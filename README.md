# Methodology: Training a K-Nearest Neighbors (KNN) Classifier AI Model to Determine Supplier Risk Levels

## Part 1: Synthetic Data Generation
First, a DataFrame was created using the filtered and renamed variables inspired by Urbaniak et al., 2022.  Their distribution can be seen in the histogram below.  Note that the population mean of several of the variables, such as Failed OTD, are shifted up to ~100 instances, and are normally distributed around this new mean, within 1 standard deviation:

![image](https://github.com/Dillon-Plummer-PhD/KNN-for-Supplier-Risk/assets/161109984/fd0a9891-4b4f-4149-8264-a80e5d1ed23d)

This difference is introduced intentionally with the following parameters:
```
SC_increase_percentage = 4
QA_increase_percentage = 1

### DataFrame Struture ###
df = pd.DataFrame({
    "% of NCMRs per Total Lots": [round(27 * QA_increase_percentage)] * 350,
    "Shipment Inaccuracy": [round(28 * SC_increase_percentage)] * 350,
    "Failed OTD": [round(24 * SC_increase_percentage)] * 350,
    "Audit Findings": [round(24 * QA_increase_percentage)] * 350,
    "Financial Obstacles": [round(24 * SC_increase_percentage)] * 350,
    "Response Delay (Docs)": [round(24 * SC_increase_percentage)] * 350,
    "Capacity Limit": [round(24 * SC_increase_percentage)] * 350,
    "Lack of Documentation (CoC, etc.)": [round(24 * QA_increase_percentage)] * 350,
    "Unjustified Price Increase": [round(24 * SC_increase_percentage)] * 350,
    "No Cost Reduction Participation": [round(24 * SC_increase_percentage)] * 350,
    "Risk Classification": [2] * 350
})

# Generate random values for columns within 1 standard deviation
for col in df.columns:
    base_value = df[col].mean()
    std = base_value * 0.1
    df[col] = [round(random.normalvariate(base_value, std)) for _ in range(350)]
```

The author was not able to obtain actual company data, due to NDAs and privacy policies, so this DataFrame represents what would be existing data from a company.  All risk classificaiton values here are set at a default of 2 (Medium Risk), and will be modified with the following code, in order to generate risk classifications for this "existing" dataset:
```
# Set thresholds for "existing" risk assessment data #
high_threshold_NCMR = 40
high_threshold_shipment = 85
high_threshold_OTD = 85
high_threshold_audit = 50
high_threshold_finance = 85
high_threshold_response_delay = 105
high_threshold_capacity = 105
high_threshold_CoCs = 45
high_threshold_price_increase = 100
high_threshold_cost_reduction = 100
# ...
# Risk Assignment #
for index, row in df.iterrows():
    high_count = 0

    # Check each column and increment counter
    for col in df.columns:
        if row[col] >= high_threshold_NCMR and col == "% of NCMRs per Total Lots":
            high_count += 1
        elif row[col] >= high_threshold_shipment and col == "Shipment Inaccuracy":
            high_count += 1
        elif row[col] >= high_threshold_OTD and col == "Failed OTD":
            high_count += 1
        elif row[col] >= high_threshold_audit and col == "Audit Findings":
            high_count += 1
        elif row[col] >= high_threshold_finance and col == "Financial Obstacles":
            high_count += 1
        elif row[col] >= high_threshold_response_delay and col == "Response Delay (Docs)":
            high_count += 1
        elif row[col] >= high_threshold_capacity and col == "Capacity Limit":
            high_count += 1
        elif row[col] >= high_threshold_CoCs and col == "Lack of Documentation (CoC, etc.)":
            high_count += 1
        elif row[col] >= high_threshold_price_increase and col == "Unjustified Price Increase":
            high_count += 1
        elif row[col] >= high_threshold_cost_reduction and col == "No Cost Reduction Participation":
            high_count += 1

    # Assign risk based on counter value
    if high_count == 3:
        df.loc[index, "Risk Classification"] = 1  # All 3 exceed
    elif high_count == 2:
        df.loc[index, "Risk Classification"] = 2  # 2 out of 3 exceed
    elif high_count == 1:
        df.loc[index, "Risk Classification"] = 2  # 1 out of 3 exceeds
    else:
        df.loc[index, "Risk Classification"] = 3  # None exceed
```

This code supplements the existing data by assigning values in the Risk Classification column based on thresholds.  This is a continuation of the synthetic data generation process.  The distribution of risk classes can be visualized in this small histogram, where 1 is Low, 2 is Medium, and 3 is High Risk:

![image](https://github.com/Dillon-Plummer-PhD/KNN-for-Supplier-Risk/assets/161109984/eaae5ce8-08ab-4cd0-8f5a-99dea0f9c4e3)

## Part 2: Training the KNN Model
The next step is to train the KNN classifier.  The data is split into training and testing data, and the KNN classifier is called from the scikit-learn library:
```
# Split the data into training and testing sets, 75/25, respectively
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Scale features using StandardScaler (on training set only)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# KNN model with prediction and evaluation (using test set labels)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_scaled, y_train)
y_pred = knn.predict(X_test_scaled)
risk_mapping = {3: "Low Risk", 2: "Medium Risk", 1: "High Risk"}
X["Risk Category"] = X["Risk Classification"].map(risk_mapping)
```

TEXT HERE
