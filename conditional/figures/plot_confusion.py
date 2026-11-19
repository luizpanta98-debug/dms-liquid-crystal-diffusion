'''
Article figure: ResNet-50 confusion matrices, real (a) beside generated (b).
The plotting code is in ../article_plots.py (shared with ordinal_analysis.py).

Usage (from conditional/):
    python figures/plot_confusion.py resnet50/logs/resnet50/test/<run>/epoch_100 out.pdf
'''
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from article_plots import plot_confusion as main  # noqa: E402

if __name__ == '__main__':
	main(sys.argv[1], sys.argv[2], layout='row')
