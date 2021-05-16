
#### 1 | Missing Data Treatment

<sub>
- Collected data we want to use in machine learning algorithms can often contain missing data <code>np.nan</code> in the  <code>feature matrix</code>. <br>
- We can use various approaches to impute/treat the rows of data in the <code>feature matrix</code>.
</sub>

<break></break>

| <sub>Example of DataFrame with missing data</sub> | <sub>Percentage of Missing Data in DataFrame</sub> |
| -- | -- |
| ![](https://images-wixmp-ed30a86b8c4ca887773594c2.wixmp.com/f/8cc1eeaa-4046-4c4a-ae93-93d656f68688/dejqh2j-5c5e309a-395b-4784-b47c-7337f2563421.png?token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ1cm46YXBwOjdlMGQxODg5ODIyNjQzNzNhNWYwZDQxNWVhMGQyNmUwIiwiaXNzIjoidXJuOmFwcDo3ZTBkMTg4OTgyMjY0MzczYTVmMGQ0MTVlYTBkMjZlMCIsIm9iaiI6W1t7InBhdGgiOiJcL2ZcLzhjYzFlZWFhLTQwNDYtNGM0YS1hZTkzLTkzZDY1NmY2ODY4OFwvZGVqcWgyai01YzVlMzA5YS0zOTViLTQ3ODQtYjQ3Yy03MzM3ZjI1NjM0MjEucG5nIn1dXSwiYXVkIjpbInVybjpzZXJ2aWNlOmZpbGUuZG93bmxvYWQiXX0.Ew1VciDi9Qheswko1Hqc-KX5xxsMLjDnlwbY8vVkiWs) | ![](https://images-wixmp-ed30a86b8c4ca887773594c2.wixmp.com/f/8cc1eeaa-4046-4c4a-ae93-93d656f68688/dejqhp6-a7d055f1-55d9-48c9-bd8c-f15f8dd0033c.png?token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ1cm46YXBwOjdlMGQxODg5ODIyNjQzNzNhNWYwZDQxNWVhMGQyNmUwIiwiaXNzIjoidXJuOmFwcDo3ZTBkMTg4OTgyMjY0MzczYTVmMGQ0MTVlYTBkMjZlMCIsIm9iaiI6W1t7InBhdGgiOiJcL2ZcLzhjYzFlZWFhLTQwNDYtNGM0YS1hZTkzLTkzZDY1NmY2ODY4OFwvZGVqcWhwNi1hN2QwNTVmMS01NWQ5LTQ4YzktYmQ4Yy1mMTVmOGRkMDAzM2MucG5nIn1dXSwiYXVkIjpbInVybjpzZXJ2aWNlOmZpbGUuZG93bmxvYWQiXX0.wse_7v05auaaWt7TvuctBkfDot5PIJg_-qdQDtqMWwU)

#### 2 | Included Approaches

<sub>
- <code>USL_SL_imputation.py</code> : Unsupervised Learning + Supervised Learning Algorithm Imputation Approach (XGB + kNN) <br>
- <code>USL_knn_imputation.py</code> : Unsupervised Learning Algorithm Imputation Approach (kNN) <br>
</sub>
