## Notes on Approach

### First Run Through
- Code is a mess; FE is manual for both train and test (very much not a pipeline)
- Need to establish what our baseline is and compare/track results against it

### Kaggle Considerations
- I suppose the test/train split isn't necessarily...since the data is already split. But then we need to rely fully on Kaggle's evaluation?

### To-dos
- [ ] Look into best practices re: training with full train set on Kaggle.
- [ ] Look into sklearn pipelines- is that the best approach to handle FE for train and test in a less manual way? How might this translate to GCP?
- [ ] ^ Confirm how we would standardize in the sklearn pipelines above.
- [ ] Review how people track/store model performance on Kaggle.