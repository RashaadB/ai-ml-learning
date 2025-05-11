from datetime import datetime
import csv


def get_expense_input():
    while True:
        date_input = input("Enter the date of the expense (YYYY-MM-DD): ")
        try:
            date = datetime.strptime(date_input, "%Y-%m-%d").date()
            break
        except ValueError:
            print("Invalid date format. Please use YYYY-MM-DD.")

    while True:
        category = input(
            "Enter the category of the expense (e.g., Food, Travel): ").strip()
        if category:
            break
        print("Category cannot be empty.")

    while True:
        amount_input = input("Enter the amount spent: $")
        try:
            amount = float(amount_input)
            if amount >= 0:
                break
            else:
                print("Amount must be non-negative.")
        except ValueError:
            print("Invalid amount. Please enter a number.")

    description = input("Enter a brief description of the expense: ").strip()

    return {
        "date": str(date),
        "category": category,
        "amount": amount,
        "description": description
    }


def display_expenses(expenses):
    if not expenses:
        print("No expenses recorded.")
        return

    print("\nStored Expenses:")
    print("-" * 60)
    for expense in expenses:
        print(f"Date: {expense['date']} | "
              f"Category: {expense['category']} | "
              f"Amount: ${expense['amount']:.2f} | "
              f"Description: {expense['description']}")
    print("-" * 60)


def main():
    expenses = []

    while True:
        print("\nAdd a new expense")
        expense = get_expense_input()
        expenses.append(expense)

        another = input(
            "Do you want to add another expense? (y/n): ").strip().lower()
        if another != 'y':
            break

    display_expenses(expenses)


def get_monthly_budget():
    while True:
        budget_input = input("Enter your monthly budget amount: ")
        try:
            budget = float(budget_input)
            if budget >= 0:
                return budget
            else:
                print("Budget must be non-negative.")
        except ValueError:
            print("Invalid input. Please enter a number.")


def evaluate_budget(expenses, budget):
    total_spent = sum(expense["amount"] for expense in expenses)
    print(f"\nTotal Spent: ${total_spent:.2f}")
    print(f"Monthly Budget: ${budget:.2f}")

    if total_spent > budget:
        print("Warning: You have exceeded your budget!")
    else:
        remaining = budget - total_spent
        print(
            f"You are within your budget. Remaining balance: ${remaining:.2f}")


def save_expenses_to_csv(expenses, filename="expenses.csv"):
    with open(filename, mode="w", newline="") as file:
        writer = csv.DictWriter(
            file, fieldnames=["date", "category", "amount", "description"])
        writer.writeheader()
        for expense in expenses:
            writer.writerow(expense)
    print(f"Expenses saved to '{filename}'")


def load_expenses_from_csv(filename="expenses.csv"):
    expenses = []
    try:
        with open(filename, mode="r", newline="") as file:
            reader = csv.DictReader(file)
            for row in reader:
                # Convert amount from string to float
                row["amount"] = float(row["amount"])
                expenses.append(row)
        print(f"Loaded {len(expenses)} expense(s) from '{filename}'")
    except FileNotFoundError:
        print(f"No previous expenses found in '{filename}'. Starting fresh.")
    return expenses


def display_menu():
    print("\n==== Personal Expense Tracker ====")
    print("1. Add Expense")
    print("2. View Expenses")
    print("3. Track Budget")
    print("4. Save Expenses")
    print("5. Exit")
    choice = input("Enter your choice (1-5): ").strip()
    return choice


def main():
    filename = "expenses.csv"
    expenses = load_expenses_from_csv(filename)
    budget = get_monthly_budget()
    print(f"\nMonthly budget set to: ${budget:.2f}")

    while True:
        choice = display_menu()

        if choice == "1":
            expense = get_expense_input()
            expenses.append(expense)
        elif choice == "2":
            display_expenses(expenses)
        elif choice == "3":
            evaluate_budget(expenses, budget)
        elif choice == "4":
            save_expenses_to_csv(expenses, filename)
        elif choice == "5":
            save_expenses_to_csv(expenses, filename)
            print("Exiting program. Goodbye!")
            break
        else:
            print("Invalid choice. Please enter a number from 1 to 5.")


if __name__ == "__main__":
    main()
