import sys;from pathlib import Path;sys.path.insert(0,str(Path(__file__).parent.parent))
from src.data import make_synthetic;from src.model import fit_and_evaluate;from src.core import profit_at_limit,constraint_satisfaction
def test_data():d=make_synthetic(500);assert d["n_samples"]==500
def test_profit():assert profit_at_limit(10000,0.04)>0
def test_fit():d=make_synthetic(500);m,met=fit_and_evaluate(d);assert met["avg_profit"]>0
if __name__=="__main__":test_data();test_profit();test_fit();print("OK")
