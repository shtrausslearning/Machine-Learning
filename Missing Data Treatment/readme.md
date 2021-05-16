
#### 1 | MISSING DATA TREATMENT

<sub>
- Collected data we want to use in machine learning algorithms can often contain missing data <code>np.nan</code> in the  <code>feature matrix</code>, example shown below. <br>
</sub>

<break></break>

| <sub>Example of DataFrame with missing data</sub> | <sub>Percentage of Missing Data in DataFrame</sub> | <sub>After Imputation DataFrame</sub> |
| -- | -- | -- |
| ![](https://images-wixmp-ed30a86b8c4ca887773594c2.wixmp.com/f/8cc1eeaa-4046-4c4a-ae93-93d656f68688/dejrfbz-704b98d3-7f31-4ac6-9721-0796d9d49c5e.png?token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ1cm46YXBwOjdlMGQxODg5ODIyNjQzNzNhNWYwZDQxNWVhMGQyNmUwIiwiaXNzIjoidXJuOmFwcDo3ZTBkMTg4OTgyMjY0MzczYTVmMGQ0MTVlYTBkMjZlMCIsIm9iaiI6W1t7InBhdGgiOiJcL2ZcLzhjYzFlZWFhLTQwNDYtNGM0YS1hZTkzLTkzZDY1NmY2ODY4OFwvZGVqcmZiei03MDRiOThkMy03ZjMxLTRhYzYtOTcyMS0wNzk2ZDlkNDljNWUucG5nIn1dXSwiYXVkIjpbInVybjpzZXJ2aWNlOmZpbGUuZG93bmxvYWQiXX0.lU6l7LqM-2ewYgxiwm8vCThpAMu4HS9JfjYJFlb-40I) | ![](https://images-wixmp-ed30a86b8c4ca887773594c2.wixmp.com/f/8cc1eeaa-4046-4c4a-ae93-93d656f68688/dejrfbb-3ecfbe51-609f-42a3-8c44-2250db13f7f4.png?token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ1cm46YXBwOjdlMGQxODg5ODIyNjQzNzNhNWYwZDQxNWVhMGQyNmUwIiwiaXNzIjoidXJuOmFwcDo3ZTBkMTg4OTgyMjY0MzczYTVmMGQ0MTVlYTBkMjZlMCIsIm9iaiI6W1t7InBhdGgiOiJcL2ZcLzhjYzFlZWFhLTQwNDYtNGM0YS1hZTkzLTkzZDY1NmY2ODY4OFwvZGVqcmZiYi0zZWNmYmU1MS02MDlmLTQyYTMtOGM0NC0yMjUwZGIxM2Y3ZjQucG5nIn1dXSwiYXVkIjpbInVybjpzZXJ2aWNlOmZpbGUuZG93bmxvYWQiXX0.oJUlzkQxJ2hlmEWSHmy7BgwykV1rKwOYh-DgqH6KSDk) | ![](https://images-wixmp-ed30a86b8c4ca887773594c2.wixmp.com/f/8cc1eeaa-4046-4c4a-ae93-93d656f68688/dejrfbq-61554b70-3464-4b36-838d-f14e378f9f46.png?token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ1cm46YXBwOjdlMGQxODg5ODIyNjQzNzNhNWYwZDQxNWVhMGQyNmUwIiwiaXNzIjoidXJuOmFwcDo3ZTBkMTg4OTgyMjY0MzczYTVmMGQ0MTVlYTBkMjZlMCIsIm9iaiI6W1t7InBhdGgiOiJcL2ZcLzhjYzFlZWFhLTQwNDYtNGM0YS1hZTkzLTkzZDY1NmY2ODY4OFwvZGVqcmZicS02MTU1NGI3MC0zNDY0LTRiMzYtODM4ZC1mMTRlMzc4ZjlmNDYucG5nIn1dXSwiYXVkIjpbInVybjpzZXJ2aWNlOmZpbGUuZG93bmxvYWQiXX0.eeC2o6m2xcPNCjFd3bWz4jtpPs-q-DCLbqfNCjQiObE)

<sub>
- We can use various approaches to impute/treat the rows of data in the <code>feature matrix</code> & a model based imputation is one such method.
- Model based approaches can often be more accurate the standard constant value pandas imputation. 
- In the current function, an <code>ensemble</code> of supervised & unsupervised learning approaches is used.
</sub>
