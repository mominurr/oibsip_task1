import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import seaborn as sns

def Classification_for_Flower_Predictions(SepalLengthCm,SepalWidthCm,PetalLengthCm,PetalWidthCm):
    # using the read_csv function in the pandas library, we load the data into a dataframe.

    df = pd.read_csv("Iris.csv")

    # clean the data
    df["SepalLengthCm"].fillna(df["SepalLengthCm"].mean(), inplace=True)
    df["SepalWidthCm"].fillna(df["SepalWidthCm"].mean(), inplace=True)
    df["PetalLengthCm"].fillna(df["PetalLengthCm"].mean(), inplace=True)
    df["PetalWidthCm"].fillna(df["PetalWidthCm"].mean(), inplace=True)
    df["Species"].fillna(df["Species"].mode()[0], inplace=True)

    # df.to_csv("iris.csv")

    # Customize the figure size using plt.figure before creating the bar plot
    plt.figure(figsize=(16, 12))  # You can adjust the width and height (e.g., 8 inches wide, 6 inches tall)

    df["Species"].value_counts().plot.bar()

    # Optionally, add labels and a title
    plt.xlabel("Species")
    plt.ylabel("Count")
    plt.title("Species Count")

    plt.savefig("iris_species_count.png")

    # seaborn pairplot saves the plot to a file
    sns.pairplot(df, hue='Species')
    plt.savefig("iris_pairplot.png")

    # define features(X) and target(Y)
    X = df[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]]
    Y = df["Species"]
    X=X.values
    Y=Y.values
    
    # Split the data into training(80%) and testing(20%)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Create a logistic regression model
    MODEL = LogisticRegression()

    # Fit the model to the training data (X_train, Y_train)
    MODEL.fit(X_train, Y_train)

    # Make predictions using the testing data
    Y_pred = MODEL.predict(X_test)

    # Calculate the accuracy of the model
    accuracy = accuracy_score(Y_test, Y_pred)
    print("\nOutput Section:\n")
    print(f'Model Accuracy: {accuracy * 100:.2f}%')

    prediction_value = MODEL.predict([[SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm]])[0]
    print(f'Predicted Flower Species : {prediction_value}')



    


if __name__ == "__main__":

    # Make predictions. Let us predict the species of a flower with SepalLengthCm = 5.4, SepalWidthCm = 2.6, PetalLengthCm = 4.1, PetalWidthCm = 1.3
    print("\n############# Welcome to the Iris Flower Classification Predictor #############\n")
    print("This program will predict the species of a flower.\nIt will give you the accuracy of the model.\nAnd also, it will give you the predicted species of the flower and give you analysis visualization images of the flower.\nALL parameters are in centimeters.\nSome examples of inputs are:\nSepalLength = 5.4 cm, SepalWidth = 2.6 cm, PetalLength = 4.1 cm, PetalWidth = 1.3 cm\nPlease follow the instructions below:\n")
    
    while True:
        print("\nChoose a one of the following options:")
        print("1. Predict the species of a flower")
        print("2. Exit")
        option = int(input("Enter your choice: "))
        if option == 1:
            print("\nInput Section: \n")
            SepalLengthCm = float(input("Enter the SepalLength (to centimeters): "))
            SepalWidthCm = float(input("Enter the SepalWidth (to centimeters): "))
            PetalLengthCm = float(input("Enter the PetalLength (to centimeters): "))
            PetalWidthCm = float(input("Enter the PetalWidth (to centimeters): "))
            Classification_for_Flower_Predictions(SepalLengthCm,SepalWidthCm,PetalLengthCm,PetalWidthCm)
        elif option == 2:
            print("Thank you for using the Flower Classification Predictor. Have a nice day!\n")
            break
        else:
            print("Please enter a valid option\n")
            continue



        