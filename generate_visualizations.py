import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create directory for visualizations if it doesn't exist
os.makedirs('static/img', exist_ok=True)

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('Set2')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

def save_plot(name):
    """Save the current plot to the static/img directory"""
    plt.tight_layout()
    plt.savefig(f'static/img/{name}.png', dpi=150, bbox_inches='tight')
    plt.close()

def main():
    print("Generating visualizations for the analysis page...")
    
    # Load dataset
    try:
        df = pd.read_csv('adult.csv')
        print(f"Dataset loaded with {len(df)} records")
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        return
    
    # 1. Income Distribution
    plt.figure(figsize=(8, 6))
    income_counts = df['income'].value_counts()
    plt.pie(income_counts, labels=income_counts.index, autopct='%1.1f%%', 
            startangle=90, colors=['#3498db', '#e74c3c'], explode=[0.05, 0.05],
            textprops={'fontsize': 14})
    plt.title('Income Distribution', fontsize=16)
    plt.legend(['≤ $50K', '> $50K'])
    save_plot('income_distribution')
    
    # 2. Gender Distribution
    plt.figure(figsize=(8, 6))
    gender_counts = df['gender'].value_counts()
    plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', 
            startangle=90, colors=['#3498db', '#e74c3c'], explode=[0.05, 0.05],
            textprops={'fontsize': 14})
    plt.title('Gender Distribution', fontsize=16)
    save_plot('gender_distribution')
    
    # 3. Age Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df['age'], bins=20, kde=True)
    plt.title('Age Distribution', fontsize=16)
    plt.xlabel('Age', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    save_plot('age_distribution')
    
    # 4. Income by Education Level
    plt.figure(figsize=(12, 8))
    education_income = pd.crosstab(df['education'], df['income'], normalize='index') * 100
    education_income = education_income.sort_values(by='>50K', ascending=False)
    education_income['>50K'].plot(kind='barh', color='#3498db')
    plt.title('Percentage of Individuals with Income >$50K by Education Level', fontsize=16)
    plt.xlabel('Percentage (%)', fontsize=14)
    plt.ylabel('Education Level', fontsize=14)
    plt.xlim(0, 100)
    plt.grid(axis='x')
    save_plot('education_income')
    
    # 5. Education Level Distribution
    plt.figure(figsize=(10, 6))
    education_counts = df['education'].value_counts()
    plt.pie(education_counts, labels=None, autopct=None, 
            startangle=90, wedgeprops={'edgecolor': 'w'})
    plt.legend(education_counts.index, loc="center left", bbox_to_anchor=(1, 0.5))
    plt.title('Education Level Distribution', fontsize=16)
    save_plot('education_distribution')
    
    # 6. Income by Occupation
    plt.figure(figsize=(12, 8))
    occupation_income = pd.crosstab(df['occupation'], df['income'], normalize='index') * 100
    occupation_income = occupation_income.sort_values(by='>50K', ascending=False)
    occupation_income['>50K'].plot(kind='barh', color='#3498db')
    plt.title('Percentage of Individuals with Income >$50K by Occupation', fontsize=16)
    plt.xlabel('Percentage (%)', fontsize=14)
    plt.ylabel('Occupation', fontsize=14)
    plt.xlim(0, 100)
    plt.grid(axis='x')
    save_plot('occupation_income')
    
    # 7. Occupation Distribution
    plt.figure(figsize=(10, 6))
    occupation_counts = df['occupation'].value_counts()
    occupation_counts.plot(kind='bar', color='#3498db')
    plt.title('Occupation Distribution', fontsize=16)
    plt.xlabel('Occupation', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    save_plot('occupation_distribution')
    
    # 8. Income by Age Group
    plt.figure(figsize=(10, 6))
    df['age_group'] = pd.cut(df['age'], bins=[16, 25, 35, 45, 55, 65, 100], 
                            labels=['<25', '25-34', '35-44', '45-54', '55-64', '65+'])
    age_income = pd.crosstab(df['age_group'], df['income'], normalize='index') * 100
    age_income['>50K'].plot(kind='bar', color='#3498db')
    plt.title('Percentage of Individuals with Income >$50K by Age Group', fontsize=16)
    plt.xlabel('Age Group', fontsize=14)
    plt.ylabel('Percentage (%)', fontsize=14)
    plt.ylim(0, 50)
    plt.grid(axis='y')
    save_plot('age_income')
    
    # 9. Gender and Income
    plt.figure(figsize=(10, 6))
    gender_income = pd.crosstab(df['gender'], df['income'], normalize='index') * 100
    gender_income['>50K'].plot(kind='bar', color='#3498db')
    plt.title('Percentage of Individuals with Income >$50K by Gender', fontsize=16)
    plt.xlabel('Gender', fontsize=14)
    plt.ylabel('Percentage (%)', fontsize=14)
    plt.ylim(0, 40)
    plt.grid(axis='y')
    save_plot('gender_income')
    
    # 10. Gender & Education vs. Income
    plt.figure(figsize=(12, 8))
    gender_edu_income = df.groupby(['gender', 'education'])['income'].apply(
        lambda x: (x == '>50K').mean() * 100).reset_index()
    gender_edu_income_pivot = pd.pivot_table(gender_edu_income, 
                                           values='income', 
                                           index='education', 
                                           columns='gender')
    gender_edu_income_pivot = gender_edu_income_pivot.sort_values(by='Male', ascending=False)
    gender_edu_income_pivot.plot(kind='barh')
    plt.title('Percentage with Income >$50K by Gender and Education', fontsize=16)
    plt.xlabel('Percentage (%)', fontsize=14)
    plt.ylabel('Education Level', fontsize=14)
    plt.legend(title='Gender')
    plt.xlim(0, 80)
    plt.grid(axis='x')
    save_plot('gender_education_income')
    
    # 11. Education vs. Hours Worked
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='education', y='hours-per-week', data=df, showfliers=False,
                order=education_income.index)
    plt.title('Hours Worked per Week by Education Level', fontsize=16)
    plt.xlabel('Education Level', fontsize=14)
    plt.ylabel('Hours per Week', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    save_plot('education_hours')
    
    # 12. Occupation by Gender
    plt.figure(figsize=(12, 8))
    occupation_gender = pd.crosstab(df['occupation'], df['gender'], normalize='index') * 100
    occupation_gender = occupation_gender.sort_values(by='Male', ascending=False)
    occupation_gender.plot(kind='barh')
    plt.title('Gender Distribution by Occupation', fontsize=16)
    plt.xlabel('Percentage (%)', fontsize=14)
    plt.ylabel('Occupation', fontsize=14)
    plt.legend(title='Gender')
    plt.xlim(0, 100)
    plt.grid(axis='x')
    save_plot('occupation_gender')
    
    # 13. Age vs. Hours Worked by Income
    plt.figure(figsize=(10, 8))
    colors = {'<=50K': '#3498db', '>50K': '#e74c3c'}
    for income, color in colors.items():
        subset = df[df['income'] == income]
        plt.scatter(subset['age'], subset['hours-per-week'], c=color, alpha=0.5, label=income)
    plt.title('Age vs. Hours Worked by Income', fontsize=16)
    plt.xlabel('Age', fontsize=14)
    plt.ylabel('Hours per Week', fontsize=14)
    plt.legend(title='Income')
    plt.grid(True)
    save_plot('age_hours_income')
    
    print("All visualizations have been generated successfully!")

if __name__ == "__main__":
    main() 