def print_list():
  print("Please select one option")
  print("1. Linear Regression")
  print("2. Logistic Regression")
  print("3. Neural Network")

print_list()
option = int(input("\nSelect any of the above option (1..3): "))
if option == 1:
  import linear_regression
elif option == 2:
  import logistic_regression
elif option == 3:
  import Neural_network