#### 1 | Unsupervised Learning Dimensionality Reduction

<sup>
- One application of Unsupervised Learning Algorithms; <code>dimensionality reduction</code>. <br>
- Often we have too many features in the feature matrix, we can reduce the number of dimension in the problem by using unsupervised learning methods. <br>
- In many problems application different approaches may work better than others, and some might not be very practical to implement due to computational cost. <br>
- In this function, most common methods available in the <code>sklearn</code> library are included & an example code is given. <br>
- An example implementation is viewable in a Kaggle Notebook; ![example](https://www.kaggle.com/shtrausslearning/building-an-asset-trading-strategy)
</sup>

| <sub>Full Feature Matrix (13 Features)</sub> | <sub>Reduced Feature Matrix (4 Features)</sub> |
| - | - |
| ![](https://images-wixmp-ed30a86b8c4ca887773594c2.wixmp.com/f/8cc1eeaa-4046-4c4a-ae93-93d656f68688/dejq952-318b4cf9-a605-486a-ad00-fc90f0bb921f.png?token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ1cm46YXBwOjdlMGQxODg5ODIyNjQzNzNhNWYwZDQxNWVhMGQyNmUwIiwiaXNzIjoidXJuOmFwcDo3ZTBkMTg4OTgyMjY0MzczYTVmMGQ0MTVlYTBkMjZlMCIsIm9iaiI6W1t7InBhdGgiOiJcL2ZcLzhjYzFlZWFhLTQwNDYtNGM0YS1hZTkzLTkzZDY1NmY2ODY4OFwvZGVqcTk1Mi0zMThiNGNmOS1hNjA1LTQ4NmEtYWQwMC1mYzkwZjBiYjkyMWYucG5nIn1dXSwiYXVkIjpbInVybjpzZXJ2aWNlOmZpbGUuZG93bmxvYWQiXX0.BiPsW7-ijQ4jiCxxwQJt1O5X_JnPMB5IuJr6vz_9pfY) | ![](https://images-wixmp-ed30a86b8c4ca887773594c2.wixmp.com/f/8cc1eeaa-4046-4c4a-ae93-93d656f68688/dejq959-f8094ae9-95f0-4fd6-a824-4a45b5b748d3.png?token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ1cm46YXBwOjdlMGQxODg5ODIyNjQzNzNhNWYwZDQxNWVhMGQyNmUwIiwiaXNzIjoidXJuOmFwcDo3ZTBkMTg4OTgyMjY0MzczYTVmMGQ0MTVlYTBkMjZlMCIsIm9iaiI6W1t7InBhdGgiOiJcL2ZcLzhjYzFlZWFhLTQwNDYtNGM0YS1hZTkzLTkzZDY1NmY2ODY4OFwvZGVqcTk1OS1mODA5NGFlOS05NWYwLTRmZDYtYTgyNC00YTQ1YjViNzQ4ZDMucG5nIn1dXSwiYXVkIjpbInVybjpzZXJ2aWNlOmZpbGUuZG93bmxvYWQiXX0.2j-oUzDCuKM_5TvrrChlRZNBidFqF7FhQZY81itAtLY) |

#### 2 | Examples of Dimensionality Reduction
<sup>
- <code>sklearn</code> has two main types of modules modules for this task; <code>decomposition</code> & <code>manifold</code>. <br>
- Some examples from <code>df_diab</code> & <code>df_boston</code> in the example code.
</sup>
<br>
  
|TNSE (Manifold) | MDS (Manifold) | PCA (Decomposition) | LLE (Manifold Learning) |
| - | - | - | - |
| ![](https://images-wixmp-ed30a86b8c4ca887773594c2.wixmp.com/f/8cc1eeaa-4046-4c4a-ae93-93d656f68688/dejpbp1-6f79b797-285f-46de-b1db-a04093b1daf7.png?token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ1cm46YXBwOjdlMGQxODg5ODIyNjQzNzNhNWYwZDQxNWVhMGQyNmUwIiwiaXNzIjoidXJuOmFwcDo3ZTBkMTg4OTgyMjY0MzczYTVmMGQ0MTVlYTBkMjZlMCIsIm9iaiI6W1t7InBhdGgiOiJcL2ZcLzhjYzFlZWFhLTQwNDYtNGM0YS1hZTkzLTkzZDY1NmY2ODY4OFwvZGVqcGJwMS02Zjc5Yjc5Ny0yODVmLTQ2ZGUtYjFkYi1hMDQwOTNiMWRhZjcucG5nIn1dXSwiYXVkIjpbInVybjpzZXJ2aWNlOmZpbGUuZG93bmxvYWQiXX0.W3yk0ED4gie1odk7VWG5IRzPSBCxR1tViGvgQoJSN5Y) | ![](https://images-wixmp-ed30a86b8c4ca887773594c2.wixmp.com/f/8cc1eeaa-4046-4c4a-ae93-93d656f68688/dejpbrb-b626aca0-e63c-486c-9feb-9b05a9cfa54e.png?token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ1cm46YXBwOjdlMGQxODg5ODIyNjQzNzNhNWYwZDQxNWVhMGQyNmUwIiwiaXNzIjoidXJuOmFwcDo3ZTBkMTg4OTgyMjY0MzczYTVmMGQ0MTVlYTBkMjZlMCIsIm9iaiI6W1t7InBhdGgiOiJcL2ZcLzhjYzFlZWFhLTQwNDYtNGM0YS1hZTkzLTkzZDY1NmY2ODY4OFwvZGVqcGJyYi1iNjI2YWNhMC1lNjNjLTQ4NmMtOWZlYi05YjA1YTljZmE1NGUucG5nIn1dXSwiYXVkIjpbInVybjpzZXJ2aWNlOmZpbGUuZG93bmxvYWQiXX0.VfyB2EI-0hUMDqiXzqGcahcygY5zKBXnSoij2qByLQ4) | ![](https://images-wixmp-ed30a86b8c4ca887773594c2.wixmp.com/f/8cc1eeaa-4046-4c4a-ae93-93d656f68688/dejpbrk-386947ca-2ad9-4753-9f19-f24018bd2f7b.png?token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ1cm46YXBwOjdlMGQxODg5ODIyNjQzNzNhNWYwZDQxNWVhMGQyNmUwIiwiaXNzIjoidXJuOmFwcDo3ZTBkMTg4OTgyMjY0MzczYTVmMGQ0MTVlYTBkMjZlMCIsIm9iaiI6W1t7InBhdGgiOiJcL2ZcLzhjYzFlZWFhLTQwNDYtNGM0YS1hZTkzLTkzZDY1NmY2ODY4OFwvZGVqcGJyay0zODY5NDdjYS0yYWQ5LTQ3NTMtOWYxOS1mMjQwMThiZDJmN2IucG5nIn1dXSwiYXVkIjpbInVybjpzZXJ2aWNlOmZpbGUuZG93bmxvYWQiXX0.8M0aiZfSuiiMsEl1RrTKRiggizZUqGjKxDlp51Rvg3Y) | ![](https://images-wixmp-ed30a86b8c4ca887773594c2.wixmp.com/f/8cc1eeaa-4046-4c4a-ae93-93d656f68688/dejpbsn-133b0cf2-fbee-488c-8843-2c78d9a0e57e.png?token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ1cm46YXBwOjdlMGQxODg5ODIyNjQzNzNhNWYwZDQxNWVhMGQyNmUwIiwiaXNzIjoidXJuOmFwcDo3ZTBkMTg4OTgyMjY0MzczYTVmMGQ0MTVlYTBkMjZlMCIsIm9iaiI6W1t7InBhdGgiOiJcL2ZcLzhjYzFlZWFhLTQwNDYtNGM0YS1hZTkzLTkzZDY1NmY2ODY4OFwvZGVqcGJzbi0xMzNiMGNmMi1mYmVlLTQ4OGMtODg0My0yYzc4ZDlhMGU1N2UucG5nIn1dXSwiYXVkIjpbInVybjpzZXJ2aWNlOmZpbGUuZG93bmxvYWQiXX0.P_epVqtFszpO0HQ1ZF8IRj2dMqioSNe9qfECE5IjAh0)
