import matplotlib.pyplot as plt
import seaborn as sns

def plot_feature_importance(model, feature_names, top=20):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):  # линейные модели
        importances = np.abs(model.coef_)
    else:
        raise ValueError('Модель не поддерживает feature importance')

    feat_imp = pd.Series(importances, index=feature_names)
    feat_imp.sort_values(ascending=False).head(top).plot(kind='barh')
    plt.title('Top {} Feature Importances'.format(top))
    plt.gca().invert_yaxis()
    plt.show()

def plot_residuals(y_true, y_pred):
    residuals = y_true - y_pred
    sns.histplot(residuals, kde=True)
    plt.xlabel('Residual')
    plt.title('Distribution of Residuals')
    plt.show()
