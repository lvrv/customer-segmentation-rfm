from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
np.random.seed(42)

n_cust = 1500
cust_ids = [f"C{str(i).zfill(5)}" for i in range(n_cust)]
start = datetime.today() - timedelta(days=365)
rows = []
for cid in cust_ids:
    n_tx = np.random.poisson(lam=5) + 1
    dates = [start + timedelta(days=int(np.random.uniform(0, 365))) for _ in range(n_tx)]
    amounts = np.round(np.random.lognormal(mean=3, sigma=0.6, size=n_tx), 2)
    for d, a in zip(dates, amounts):
        rows.append((cid, d, a))

tx = pd.DataFrame(rows, columns=["customer_id", "date", "amount"])

snapshot = tx["date"].max() + pd.Timedelta(days=1)
rfm = (tx.groupby("customer_id")
         .agg(Recency=("date", lambda s: (snapshot - s.max()).days),
              Frequency=("customer_id", "size"),
              Monetary=("amount", "sum"))
         .reset_index())

rfm_log = rfm.copy()
rfm_log[["Recency", "Frequency", "Monetary"]] = np.log1p(rfm_log[["Recency", "Frequency", "Monetary"]])

scaler = StandardScaler()
X = scaler.fit_transform(rfm_log[["Recency", "Frequency", "Monetary"]])

best_k, best_score, best_model = None, -1, None
for k in range(2, 7):
    km = KMeans(n_clusters=k, random_state=42, n_init="auto")
    labels = km.fit_predict(X)
    s = silhouette_score(X, labels)
    if s > best_score:
        best_k, best_score, best_model = k, s, km

labels = best_model.predict(X)
rfm["segment"] = labels

profiles = (rfm.groupby("segment")
              .agg(customers=("customer_id", "nunique"),
                   Recency_median=("Recency", "median"),
                   Frequency_median=("Frequency", "median"),
                   Monetary_median=("Monetary", "median"),
                   Monetary_sum=("Monetary", "sum"))
              .sort_values("Monetary_sum", ascending=False))

sns.set(style="whitegrid")
plt.figure(figsize=(7, 5))
sns.scatterplot(data=rfm, x="Recency", y="Monetary", hue="segment", alpha=0.7)
plt.title(f"R vs M по сегментам (k={best_k}, silhouette={best_score:.2f})")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "scatter_R_M.png", dpi=140)
plt.close()

rfm.to_csv(OUTPUT_DIR / "rfm_with_segments.csv", index=False)
profiles.to_csv(OUTPUT_DIR / "segment_profiles.csv")

print(f"✅ Сегментация завершена (k={best_k}, silhouette={best_score:.3f}). Артефакты: {OUTPUT_DIR}")