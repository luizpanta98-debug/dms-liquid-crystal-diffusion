import torch as th


class DM:
	def __init__(self, beta_1=1e-4, beta_T=0.02, T=1000, prediction_type='x0', guidance_scale=1.0):
		if prediction_type not in ('eps', 'x0'):
			raise ValueError('prediction_type must be x0 or eps.')
		self.prediction_type = prediction_type
		self.guidance_scale = guidance_scale
		self.beta_1 = beta_1
		self.beta_T = beta_T
		self.T = T
		self.beta = th.linspace(beta_1, beta_T, T)
		self.alpha = 1 - self.beta
		self.alpha_bar = self.alpha.cumprod(-1)

	def q(self, x_0, t):
		'''
		x_0: input data minibatch
		t: timesteps minibatch

		x_t ~ q(x_t|x_0) = N(x_t; sqrt(alpha_bar_t)*x_0, (1-alpha_bar_t)I )
		x_t = sqrt(alpha_bar_t)*x_0 + sqrt(1-alpha_bar_t)*eps

		Return:
			x_t, eps
		'''
		dev = t.device
		eps = th.randn_like(x_0)
		alpha_bar_t = self.alpha_bar.to(dev).gather(-1,t).view(-1,1,1,1)
		x_t = alpha_bar_t.sqrt()*x_0 + (1-alpha_bar_t).sqrt() * eps

		return x_t, eps

	def p(self, x_t, t, y, net):
		'''
		x_t: input data minibatch
		t: FIXED timestep minibatch
		net: Neural Network conditioned on x_t, t and y

		x_{t-1} ~ p(x{t-1}|x_t) = N(x_t; mu_theta, beta_t * I )
		mu_theta = (1/sqrt(alpha_t))*(x_t - beta_t * eps / sqrt(1-alpha_bar_t))
		z = torch.randn_like(x_t) if t[0] >= 1 else 0
		x_{t-1} = mu_theta + sqrt(beta_t) * z

		Return:
			x_t
		'''
		dev = t.device
		z = th.randn_like(x_t) if t[0] >= 1 else 0
		alpha_bar_t = self.alpha_bar.to(dev).gather(-1,t).view(-1,1,1,1)
		alpha_t = self.alpha.to(dev).gather(-1,t).view(-1,1,1,1)
		beta_t = self.beta.to(dev).gather(-1,t).view(-1,1,1,1)
		_, eps = self.ddim_pred_x0(net, x_t, t, y)

		mu_theta = (1/th.sqrt(alpha_t))*(x_t - beta_t * eps / th.sqrt(1-alpha_bar_t))
		x_t_prev = mu_theta + th.sqrt(beta_t) * z
		return x_t_prev

	def ddim_pred_x0(self, net, x_t, t, y, guidance_scale=None):
		dev=x_t.device
		alpha_bar_t = self.alpha_bar.to(dev).gather(-1,t).view(-1,1,1,1)
		scale = self.guidance_scale if guidance_scale is None else guidance_scale
		if scale != 1 and y is not None:
			# Classifier-free guidance: conditional and unconditional (null label = zeros)
			# predictions in one batch of size 2B, then
			# pred = uncond + scale * (cond - uncond).
			both = net(th.cat([x_t, x_t]), th.cat([t, t]), th.cat([y, th.zeros_like(y)]))
			cond, uncond = both.float().chunk(2)
			pred = uncond + scale * (cond - uncond)
		else:
			pred = net(x_t, t, y).float()
		if self.prediction_type == 'x0':
			sample = pred
			eps_pred = (x_t - alpha_bar_t.sqrt()*sample) / (1-alpha_bar_t).sqrt()
		elif self.prediction_type == 'eps':
			eps_pred = pred
			sample = (x_t - (1-alpha_bar_t).sqrt() * eps_pred) / alpha_bar_t.sqrt()
		return sample, eps_pred

	def ddim_sample(self, eta, net, x_t, t, t_prev, y):
		dev = x_t.device
		alpha_bar_t = self.alpha_bar.to(dev).gather(-1, t).view(-1,1,1,1)
		alpha_bar_t_minus_one = self.alpha_bar.to(dev).gather(-1, t_prev).view(-1,1,1,1)
		sigma_t = eta * (((1 - alpha_bar_t_minus_one) / (1 - alpha_bar_t)).sqrt() * (1 - alpha_bar_t/alpha_bar_t_minus_one).sqrt())
		x_0, eps_pred = self.ddim_pred_x0(net, x_t, t, y)
		x_0 = th.clamp(x_0, -1, 1) # Enforce x_0 to be in data domain [-1,1]
		x_prev = alpha_bar_t_minus_one.sqrt() * x_0 + (1 - alpha_bar_t_minus_one - sigma_t**2).sqrt() * eps_pred + sigma_t * th.randn_like(x_t)
		return x_prev, x_0, eps_pred

	def _normal_cdf(self, x):
		u = (2/th.pi)**0.5 * (x + 0.044715*x**3)
		return (1 + u.tanh()) / 2

	def _decoder_nll(self, x_0, mu_theta, beta_t):
		'''
		Discretized Gaussian NLL for uint8 images mapped to [-1,1].
		Return: bits/dim per image.
		'''
		std = beta_t.sqrt()
		cdf_lower = self._normal_cdf((x_0-mu_theta-1/255) / std)
		cdf_upper = self._normal_cdf((x_0-mu_theta+1/255) / std)

		prob = cdf_upper - cdf_lower
		prob = th.where(x_0 < -0.999, cdf_upper, prob)
		prob = th.where(x_0 > 0.999, 1-cdf_lower, prob)

		log_2 = x_0.new_tensor(2.).log()
		return -prob.clamp(min=1e-12).log().mean((1,2,3)) / log_2

	@th.no_grad()
	def nll(self, x_0, y, net, steps=None):
		'''
		Estimate the variational NLL bound in bits/dim.
		Use net.eval() and uint8 images normalized to [-1,1].

		steps: None for the exact bound (all T terms), or a list of
			(timestep index, weight) pairs for a Monte Carlo estimate of the
			sum of the terms (see stratified_steps).

		Return:
			total_bpd: (B,), total bound per image
			prior_bpd: (B,), terminal KL per image
			vb: (B,len(steps)), weighted terms in the order of steps
				(t=T-1,...,0 for the exact bound)
		'''
		if steps is None:
			steps = [(i, 1.0) for i in reversed(range(self.T))]
		dev = x_0.device
		beta = self.beta.to(dev)
		alpha = self.alpha.to(dev)
		alpha_bar = self.alpha_bar.to(dev)
		alpha_bar_prev = th.cat((th.ones_like(alpha_bar[:1]), alpha_bar[:-1]))
		log_2 = x_0.new_tensor(2.).log()

		vb = []
		for i, weight in steps:
			t = th.full((x_0.shape[0],), i, device=dev, dtype=th.long)
			x_t, _ = self.q(x_0, t)
			_, eps = self.ddim_pred_x0(net, x_t, t, y, guidance_scale=1.0)

			# Same reverse mean and variance as p().
			mu_theta = (
				x_t - beta[i]*eps / (1-alpha_bar[i]).sqrt()
			) / alpha[i].sqrt()

			if i == 0:
				term = self._decoder_nll(x_0, mu_theta, beta[i])
			else:
				# q(x_{t-1}|x_t,x_0).
				var_q = beta[i]*(1-alpha_bar_prev[i]) / (1-alpha_bar[i])
				mu_q = (
					beta[i]*alpha_bar_prev[i].sqrt()*x_0
					+ alpha[i].sqrt()*(1-alpha_bar_prev[i])*x_t
				) / (1-alpha_bar[i])

				kl = 0.5 * (
					(beta[i]/var_q).log()
					+ (var_q + (mu_q-mu_theta)**2) / beta[i]
					- 1
				)
				term = kl.mean((1,2,3)) / log_2

			vb.append(weight * term)

		# KL(q(x_T|x_0) || N(0,I)).
		prior = 0.5 * (
			alpha_bar[-1]*x_0**2
			- alpha_bar[-1]
			- (1-alpha_bar[-1]).log()
		)
		prior_bpd = prior.mean((1,2,3)) / log_2

		vb = th.stack(vb, dim=1)
		return {
			'total_bpd': vb.sum(1) + prior_bpd,
			'prior_bpd': prior_bpd,
			'vb': vb,
		}


def stratified_steps(T, num_steps, exact, rng):
	'''
	Timesteps for an unbiased Monte Carlo estimate of the sum of the T terms.

	The first `exact` terms (t = 0, ..., exact-1), which carry most of the
	bound, are computed exactly. The remaining range is split into
	`num_steps` strata of (almost) equal size; one timestep is drawn
	uniformly in each stratum and weighted by the stratum size.
	Returns a list of (timestep index, weight).
	'''
	steps = [(i, 1.0) for i in range(exact)]
	edges = [exact + round(k*(T-exact)/num_steps) for k in range(num_steps+1)]
	for lo, hi in zip(edges[:-1], edges[1:]):
		if hi > lo:
			steps.append((int(rng.integers(lo, hi)), float(hi-lo)))
	return steps
