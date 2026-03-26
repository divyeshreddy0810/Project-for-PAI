"""Pre-trained HMM regime detector using full 22-year EUR/USD history."""
import os, pickle, warnings
from datetime import datetime
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM
warnings.filterwarnings('ignore', category=UserWarning)

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
_MODEL_PATH = os.path.join(_PROJECT_ROOT, "data", "models", "hmm_pretrained.pkl")

class PretrainedHMMRegime:
    RETRAIN_DAYS = 30
    def __init__(self, n_states=3, n_iter=1000):
        self.n_states=n_states; self.n_iter=n_iter
        self.model=None; self.scaler=StandardScaler()
        self.training_date=None; self.feature_names=None

    def _engineer_features(self, df):
        df=df.copy()
        df["daily_return"]=df["Close"].pct_change()
        df["log_vol_10d"]=np.log(df["daily_return"].rolling(10).std().replace(0,np.nan).fillna(1e-8)+1e-8)
        delta=df["Close"].diff()
        gain=delta.clip(lower=0).rolling(14).mean()
        loss=(-delta.clip(upper=0)).rolling(14).mean()
        df["RSI_norm"]=(100-(100/(1+gain/(loss+1e-8))))/100.0
        ema12=df["Close"].ewm(span=12).mean(); ema26=df["Close"].ewm(span=26).mean()
        df["MACD_norm"]=(ema12-ema26)/(df["Close"]+1e-8)
        df["momentum_10d"]=df["Close"].pct_change(10)
        return df[["daily_return","log_vol_10d","RSI_norm","MACD_norm","momentum_10d"]].dropna()

    def _fetch_training_data(self):
        print("  📊 Fetching 22-year EUR/USD history for HMM training...")
        df=yf.download("EURUSD=X",start="2004-01-01",end=datetime.now().strftime("%Y-%m-%d"),progress=False)
        if isinstance(df.columns,pd.MultiIndex): df.columns=df.columns.droplevel(1)
        print(f"  ✅ {len(df)} rows ({df.index[0].date()} → {df.index[-1].date()})")
        return df

    def train(self, force=False):
        if not force and self._model_fresh():
            print("  ✅ Using existing pre-trained HMM"); return self.load()
        print("  🔄 Training HMM on 22 years of data...")
        df=self._fetch_training_data(); features=self._engineer_features(df)
        if len(features)<1000: raise ValueError(f"Too few rows: {len(features)}")
        X=self.scaler.fit_transform(features); self.feature_names=list(features.columns)
        self.model=GaussianHMM(n_components=self.n_states,covariance_type="diag",
                               n_iter=self.n_iter,random_state=42,tol=1e-4)
        self.model.fit(X); self.training_date=datetime.now(); self._save()
        print(f"  ✅ HMM trained on {len(features)} rows"); return self

    def _save(self):
        os.makedirs(os.path.dirname(_MODEL_PATH),exist_ok=True)
        with open(_MODEL_PATH,"wb") as f:
            pickle.dump({"model":self.model,"scaler":self.scaler,
                         "feature_names":self.feature_names,
                         "training_date":self.training_date,"n_states":self.n_states},f)
        print(f"  💾 Saved → {_MODEL_PATH}")

    def load(self):
        try:
            with open(_MODEL_PATH,"rb") as f: state=pickle.load(f)
            self.model=state["model"]; self.scaler=state["scaler"]
            self.feature_names=state["feature_names"]; self.training_date=state["training_date"]
            print(f"  📂 Loaded HMM (trained {self.training_date.strftime('%Y-%m-%d')})"); return self
        except Exception as e:
            print(f"  ⚠️  Load failed: {e}"); return None

    def _model_fresh(self):
        if not os.path.exists(_MODEL_PATH): return False
        try:
            with open(_MODEL_PATH,"rb") as f: state=pickle.load(f)
            td=state.get("training_date")
            return td is not None and (datetime.now()-td).days<self.RETRAIN_DAYS
        except: return False

    def is_stale(self): return not self._model_fresh()

    def predict(self, df):
        if self.model is None:
            if not self.load(): return "sideways"
        try:
            features=self._engineer_features(df)
            if len(features)<10: return "sideways"
            recent=features.tail(20); X=self.scaler.transform(recent)
            states=self.model.predict(X)
            returns=recent["daily_return"].values
            state_ret={s: returns[states==s].mean() if (states==s).sum()>0 else 0.0
                       for s in range(self.n_states)}
            sorted_s=sorted(state_ret,key=state_ret.get)
            regime_map={sorted_s[0]:"bear",sorted_s[-1]:"bull"}
            for s in sorted_s[1:-1]: regime_map[s]="sideways"
            current=int(pd.Series(states[-5:]).mode()[0])
            return regime_map.get(current,"sideways")
        except Exception as e:
            print(f"  ⚠️  HMM predict failed: {e}"); return "sideways"

_instance=None
def get_pretrained_hmm():
    global _instance
    if _instance is None:
        _instance=PretrainedHMMRegime()
        if _instance.is_stale(): _instance.train()
        else: _instance.load()
    return _instance
