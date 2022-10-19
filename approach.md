## Notes on Approach

### First Run Through
- Code is a mess; Preprocessing and FE is manual for both train and test (very much not a pipeline)
- Need to establish what our baseline is and compare/track results against it
- Try random forest, but then potentially start small (feature-wise) and build upwards.

### Kaggle Considerations
- I suppose the test/train split isn't necessarily...since the data is already split. But then we need to rely fully on Kaggle's evaluation?

### To-dos
- [X] Look into best practices re: training with full train set on Kaggle.
    - I think we would consider the train/test derived from the training data to be train/validation. (https://datascience.stackexchange.com/questions/33008/is-it-always-better-to-use-the-whole-dataset-to-train-the-final-model)
    - Use training/validation to determine which model architecture/hyperparameters should be used, THEN retrain that model with those hyperparameters on the full train/validation set and use that to make predictions on the test set.
- [X] Look into sklearn pipelines- is that the best approach to handle FE for train and test in a less manual way? How might this translate to GCP?
    - A really useful overview of sklearn pipelines: https://towardsdatascience.com/how-to-use-sklearn-pipelines-for-ridiculously-neat-code-a61ab66ca90d
    - ^ Can't stress enough how useful this is
- [X] ^ Confirm how we would standardize in the sklearn pipelines above.
    - https://stackoverflow.com/questions/54034991/using-standardization-in-sklearn-pipeline
- [X] Review how people track/store model performance on Kaggle.

### Things to implement
- [X] For one hot encoding, use handle_unknown=False to ignore new categories.
- [X] Create json (or csv) file to store model results/notes.

### Modeling Approach
- [X] Build a baseline model- LR with total SF as the independent.
- [ ] Deep dive on linear regression (~1 week)
    - [ ] Build out FE classes to use with LR and other algorithms.
    - [ ] Build out best attempt at LR with new features.
    - [ ] Capture/discuss assumptions.
    - [ ] Review coefficient p-values.
    - [ ] Review relevant metrics.
- [ ] Deep dive on bagging and boosting.
    - [ ] Try different algorithms.
    - [ ] Understand the differences between them.
    - [ ] Use grid search for hyperparameter tuning.