import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Loan Default Risk Analysis - EDA")

# Load Dataset
df = pd.read_csv("Loan_default.csv")

st.subheader("Dataset Preview")
st.write(df.head())

# 2. Loan Amount Distribution
st.subheader("Loan Amount Distribution")
fig, ax = plt.subplots()
sns.histplot(df['LoanAmount'], bins=30, color="steelblue", ax=ax)
ax.set_title("Loan Amount Distribution")
ax.set_xlabel("Loan Amount")
ax.set_ylabel("Frequency")
st.pyplot(fig)

# 3. Income Distribution
st.subheader("Income Distribution")
fig, ax = plt.subplots()
sns.histplot(df['Income'], bins=30, color="steelblue", ax=ax)
ax.set_title("Income Distribution")
ax.set_xlabel("Income")
ax.set_ylabel("Frequency")
st.pyplot(fig)

# 4. Income vs Default
st.subheader("Income vs Loan Default")
fig, ax = plt.subplots()
sns.boxplot(x='Default', y='Income', data=df, color="skyblue", ax=ax)
ax.set_title("Income vs Loan Default")
st.pyplot(fig)

# 5. Debt-To-Income Ratio vs Default
st.subheader("Debt-To-Income Ratio vs Default")
fig, ax = plt.subplots()
sns.boxplot(x='Default', y='DTIRatio', data=df, color="skyblue", ax=ax)
ax.set_title("Debt-To-Income Ratio vs Default")
st.pyplot(fig)

# 6. Interest Rate vs Default
st.subheader("Interest Rate vs Default")
fig, ax = plt.subplots()
sns.boxplot(x='Default', y='InterestRate', data=df, color="skyblue", ax=ax)
ax.set_title("Interest Rate vs Default")
st.pyplot(fig)

# 7. Feature Engineering
df['Income_to_Loan_Ratio'] = df['Income'] / df['LoanAmount']
st.subheader("New Feature Created: Income_to_Loan_Ratio")
st.write(df[['Income','LoanAmount','Income_to_Loan_Ratio']].head())

# 8. Risk Segmentation
df['risk_level'] = np.where(df['DTIRatio'] > 20, "High Risk", "Low Risk")

st.subheader("Risk Level Distribution")
st.write(df['risk_level'].value_counts())

fig, ax = plt.subplots()
sns.countplot(x='risk_level', data=df, color="steelblue", ax=ax)
ax.set_title("Borrower Risk Segmentation")
st.pyplot(fig)

# 9. Correlation Heatmap
st.subheader("Feature Correlation Heatmap")
fig, ax = plt.subplots(figsize=(10,8))
sns.heatmap(df.corr(numeric_only=True), cmap="Blues", annot=False, ax=ax)
st.pyplot(fig)

# 10. Income vs Loan Amount
st.subheader("Income vs Loan Amount by Default")
fig, ax = plt.subplots()
sns.scatterplot(x='Income', y='LoanAmount', hue='Default', data=df, ax=ax)
st.pyplot(fig)

# 11. Pairplot
st.subheader("Pairplot of Important Features")

important_features = ['LoanAmount','Income','DTIRatio','CreditScore']
existing_features = [f for f in important_features if f in df.columns]

if len(existing_features) > 1:
    pairplot = sns.pairplot(df[existing_features])
    st.pyplot(pairplot)

st.subheader("Conclusion")

st.write("""
1. Borrowers with higher Debt-To-Income (DTI) ratios show a greater likelihood of loan default.
2. Income levels influence repayment capacity, but loan amount relative to income is also an important factor.
3. Interest rates tend to be higher for borrowers who fall into higher risk categories.
4. The Income-to-Loan ratio helps indicate whether a borrower has sufficient income to manage the loan amount.
5. Risk segmentation based on DTI ratio highlights borrowers who may require stricter credit evaluation.

Overall, the exploratory data analysis helps identify important financial indicators that influence loan default risk and can support better credit risk assessment.
""")