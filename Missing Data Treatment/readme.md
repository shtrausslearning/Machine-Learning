
#### 1 | MISSING DATA TREATMENT

<sub>
- Collected data we want to use in machine learning algorithms can often contain missing data <code>np.nan</code> in the  <code>feature matrix</code>, example shown below. <br>
</sub>

<break></break>

| <sub>Example of DataFrame with missing data</sub> | <sub>Percentage of Missing Data in DataFrame</sub> |
| -- | -- |
| ![](https://images-wixmp-ed30a86b8c4ca887773594c2.wixmp.com/f/8cc1eeaa-4046-4c4a-ae93-93d656f68688/dejrfbz-704b98d3-7f31-4ac6-9721-0796d9d49c5e.png?token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ1cm46YXBwOjdlMGQxODg5ODIyNjQzNzNhNWYwZDQxNWVhMGQyNmUwIiwiaXNzIjoidXJuOmFwcDo3ZTBkMTg4OTgyMjY0MzczYTVmMGQ0MTVlYTBkMjZlMCIsIm9iaiI6W1t7InBhdGgiOiJcL2ZcLzhjYzFlZWFhLTQwNDYtNGM0YS1hZTkzLTkzZDY1NmY2ODY4OFwvZGVqcmZiei03MDRiOThkMy03ZjMxLTRhYzYtOTcyMS0wNzk2ZDlkNDljNWUucG5nIn1dXSwiYXVkIjpbInVybjpzZXJ2aWNlOmZpbGUuZG93bmxvYWQiXX0.lU6l7LqM-2ewYgxiwm8vCThpAMu4HS9JfjYJFlb-40I) | ![](https://images-wixmp-ed30a86b8c4ca887773594c2.wixmp.com/f/8cc1eeaa-4046-4c4a-ae93-93d656f68688/dejqhp6-a7d055f1-55d9-48c9-bd8c-f15f8dd0033c.png?token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ1cm46YXBwOjdlMGQxODg5ODIyNjQzNzNhNWYwZDQxNWVhMGQyNmUwIiwiaXNzIjoidXJuOmFwcDo3ZTBkMTg4OTgyMjY0MzczYTVmMGQ0MTVlYTBkMjZlMCIsIm9iaiI6W1t7InBhdGgiOiJcL2ZcLzhjYzFlZWFhLTQwNDYtNGM0YS1hZTkzLTkzZDY1NmY2ODY4OFwvZGVqcWhwNi1hN2QwNTVmMS01NWQ5LTQ4YzktYmQ4Yy1mMTVmOGRkMDAzM2MucG5nIn1dXSwiYXVkIjpbInVybjpzZXJ2aWNlOmZpbGUuZG93bmxvYWQiXX0.wse_7v05auaaWt7TvuctBkfDot5PIJg_-qdQDtqMWwU)

<sub>
- We can use various approaches to impute/treat the rows of data in the <code>feature matrix</code> & a model based imputation is one such method.
- Model based approaches can often be more accurate the standard constant value pandas imputation. 
- In the current function, an <code>ensemble</code> of supervised & unsupervised learning approaches is used.
</sub>
