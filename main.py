import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

data = pd.read_csv("Concrete_Data.csv")

test_data = data.iloc[500:630].copy()
train_data = pd.concat([data.iloc[:500], data.iloc[630:]]).copy()

train_x = train_data.iloc[:, :-1].to_numpy()
train_y = train_data.iloc[:, -1].to_numpy()

test_x = test_data.iloc[:, :-1].to_numpy()
test_y = test_data.iloc[:, -1].to_numpy()

feature_names = [
    "Cement",
    "Blast Furnace Slag",
    "Fly Ash",
    "Water",
    "Superplasticizer",
    "Coarse Aggregate",
    "Fine Aggregate",
    "Age"
]


def get_mse(actual, predicted):
    return np.mean((actual - predicted) ** 2)


def get_r2(actual, predicted):
    return 1 - get_mse(actual, predicted) / np.var(actual)


def gradient_descent(x, y, learning_rate, iterations):
    rows, columns = x.shape

    weights = np.zeros(columns)
    bias = 0.0
    losses = []

    for i in range(iterations):
        predictions = x @ weights + bias
        errors = predictions - y

        weight_gradient = (2 / rows) * (x.T @ errors)
        bias_gradient = (2 / rows) * np.sum(errors)

        weights = weights - learning_rate * weight_gradient
        bias = bias - learning_rate * bias_gradient

        losses.append(get_mse(y, predictions))

    return weights, bias, losses


original_weights, original_bias, original_losses = gradient_descent(
    train_x, train_y, 0.0000001, 10000
)

scaler = StandardScaler()
scaler.fit(train_x)

scaled_train_x = scaler.transform(train_x)
scaled_test_x = scaler.transform(test_x)

scaled_weights, scaled_bias, scaled_losses = gradient_descent(
    scaled_train_x, train_y, 0.01, 10000
)

plt.figure()
plt.plot(original_losses, label="Untransformed, LR = 0.0000001")
plt.plot(scaled_losses, label="Standardized, LR = 0.01")
plt.xlabel("Iteration")
plt.ylabel("MSE")
plt.legend()
plt.tight_layout()
plt.savefig("Loss Graph 1.png")
plt.close()

log_train_x = np.log1p(train_x)
log_test_x = np.log1p(test_x)

log_weights, log_bias, log_losses = gradient_descent(
    log_train_x, train_y, 0.001, 10000
)

plt.figure()
plt.plot(original_losses, label="Untransformed, LR = 0.0000001")
plt.plot(scaled_losses, label="Standardized, LR = 0.01")
plt.plot(log_losses, label="Log-Transformed, LR = 0.001")
plt.xlabel("Iteration")
plt.ylabel("MSE")
plt.legend()
plt.tight_layout()
plt.savefig("Loss Graph 2.png")
plt.close()

original_model = LinearRegression()
original_model.fit(train_x, train_y)
original_train_pred = original_model.predict(train_x)
original_test_pred = original_model.predict(test_x)

scaled_model = LinearRegression()
scaled_model.fit(scaled_train_x, train_y)
scaled_train_pred = scaled_model.predict(scaled_train_x)
scaled_test_pred = scaled_model.predict(scaled_test_x)

log_model = LinearRegression()
log_model.fit(log_train_x, train_y)
log_train_pred = log_model.predict(log_train_x)
log_test_pred = log_model.predict(log_test_x)

model_results = pd.DataFrame({
    "Model": ["Untransformed", "Standardized", "Log-Transformed"],
    "Training MSE": [
        get_mse(train_y, original_train_pred),
        get_mse(train_y, scaled_train_pred),
        get_mse(train_y, log_train_pred)
    ],
    "Testing MSE": [
        get_mse(test_y, original_test_pred),
        get_mse(test_y, scaled_test_pred),
        get_mse(test_y, log_test_pred)
    ],
    "Training R2": [
        get_r2(train_y, original_train_pred),
        get_r2(train_y, scaled_train_pred),
        get_r2(train_y, log_train_pred)
    ],
    "Testing R2": [
        get_r2(test_y, original_test_pred),
        get_r2(test_y, scaled_test_pred),
        get_r2(test_y, log_test_pred)
    ]
})

learning_rates = [
    0.000001,
    0.000001,
    0.000001,
    0.000001,
    0.000001,
    0.000001,
    0.000001,
    0.00001
]

iteration_counts = [
    10000,
    200000,
    200000,
    200000,
    200000,
    200000,
    200000,
    200000
]

single_results = []
single_coefficients = []
multiple_coefficients = []

for i in range(8):
    one_x = train_x[:, i:i + 1]

    weights, bias, losses = gradient_descent(
        one_x,
        train_y,
        learning_rates[i],
        iteration_counts[i]
    )

    predictions = one_x @ weights + bias

    single_results.append({
        "Predictor": feature_names[i],
        "Learning Rate": learning_rates[i],
        "Iterations": iteration_counts[i],
        "MSE": get_mse(train_y, predictions),
        "R2": get_r2(train_y, predictions)
    })

    uni_model = LinearRegression()
    uni_model.fit(one_x, train_y)
    single_coefficients.append(uni_model.coef_[0])
    multiple_coefficients.append(original_model.coef_[i])

single_results_df = pd.DataFrame(single_results)

print("Univariate Gradient Descent Results")
print(single_results_df.round(4).to_string(index=False))

coefficient_results = pd.DataFrame({
    "Predictor": feature_names,
    "Univariate Coefficient": single_coefficients,
    "Multivariate Coefficient": multiple_coefficients
})

print("Coefficient Comparison")
print(coefficient_results.round(4).to_string(index=False))

fig, ax = plt.subplots(figsize=(7, 3.2))
ax.axis("off")
tbl = ax.table(
    cellText=coefficient_results.round(4).values.tolist(),
    colLabels=list(coefficient_results.columns),
    loc="center",
    cellLoc="center"
)
tbl.scale(1, 1.8)
plt.tight_layout()
plt.savefig("coefficient_comparison_table.png", bbox_inches="tight")
plt.close()

original_x_ols = sm.add_constant(train_x)
original_ols = sm.OLS(train_y, original_x_ols).fit()
original_pvalues = original_ols.pvalues[1:]

scaled_x_ols = sm.add_constant(scaled_train_x)
scaled_ols = sm.OLS(train_y, scaled_x_ols).fit()
scaled_pvalues = scaled_ols.pvalues[1:]

log_x_ols = sm.add_constant(log_train_x)
log_ols = sm.OLS(train_y, log_x_ols).fit()
log_pvalues = log_ols.pvalues[1:]

pvalue_rows = []
for i in range(8):
    row = [feature_names[i]]
    for val in [original_pvalues[i], scaled_pvalues[i], log_pvalues[i]]:
        if val < 0.001:
            row.append(f"{val:.2e}")
        else:
            row.append(f"{val:.4f}")
    pvalue_rows.append(row)

pvalue_results = pd.DataFrame(
    pvalue_rows,
    columns=[
        "Predictor",
        "Untransformed p-value",
        "Standardized p-value",
        "Log-Transformed p-value"
    ]
)

print("P-Value Comparison")
print(pvalue_results.to_string(index=False))

fig, ax = plt.subplots(figsize=(8, 3.2))
ax.axis("off")
tbl = ax.table(
    cellText=pvalue_rows,
    colLabels=list(pvalue_results.columns),
    loc="center",
    cellLoc="center"
)
tbl.scale(1, 1.8)
plt.tight_layout()
plt.savefig("pvalue_comparison_table.png", bbox_inches="tight")
plt.close()