import json
import os
import hashlib

USERS_FILE = "users.json"
TASKS_FILE = "tasks.json"

# --- Password hashing ---


def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# --- Load users from file ---


def load_users():
    if not os.path.exists(USERS_FILE):
        return {}
    with open(USERS_FILE, "r") as file:
        return json.load(file)

# --- Save users to file ---


def save_users(users):
    with open(USERS_FILE, "w") as file:
        json.dump(users, file)

# --- Register a new user ---


def register_user():
    users = load_users()
    print("\n--- User Registration ---")
    while True:
        username = input("Enter a new username: ").strip()
        if username in users:
            print("Username already exists. Please choose another.")
        else:
            break
    password = input("Enter a new password: ").strip()
    hashed_pw = hash_password(password)
    users[username] = hashed_pw
    save_users(users)
    print(f"Registration successful. You can now log in, {username}.")

# --- Login user ---


def login_user():
    users = load_users()
    print("\n--- User Login ---")
    username = input("Username: ").strip()
    password = input("Password: ").strip()
    hashed_input = hash_password(password)

    if users.get(username) == hashed_input:
        print(f"Login successful. Welcome, {username}!")
        return username
    else:
        print("Invalid username or password.")
        return None

# --- Load tasks from file ---


def load_tasks():
    if not os.path.exists(TASKS_FILE):
        return {}
    with open(TASKS_FILE, "r") as file:
        return json.load(file)

# --- Save tasks to file ---


def save_tasks(tasks):
    with open(TASKS_FILE, "w") as file:
        json.dump(tasks, file)

# --- Add a new task ---


def add_task(username):
    tasks = load_tasks()
    user_tasks = tasks.get(username, [])

    print("\n--- Add New Task ---")
    description = input("Enter task description: ").strip()
    if not description:
        print("Task description cannot be empty.")
        return

    next_id = max([task["id"] for task in user_tasks], default=0) + 1

    new_task = {
        "id": next_id,
        "description": description,
        "status": "Pending"
    }

    user_tasks.append(new_task)
    tasks[username] = user_tasks
    save_tasks(tasks)

    print(f"Task added successfully with ID {next_id}.")

# --- View tasks ---


def view_tasks(username):
    tasks = load_tasks()
    user_tasks = tasks.get(username, [])

    print(f"\n--- Tasks for {username} ---")
    if not user_tasks:
        print("No tasks found.")
        return

    for task in user_tasks:
        print(
            f"ID: {task['id']} | Description: {task['description']} | Status: {task['status']}")

# --- Mark task as completed ---


def mark_task_completed(username):
    tasks = load_tasks()
    user_tasks = tasks.get(username, [])

    if not user_tasks:
        print("No tasks to update.")
        return

    print("\n--- Mark Task as Completed ---")
    try:
        task_id = int(input("Enter the task ID to mark as completed: "))
    except ValueError:
        print("Invalid input. Please enter a numeric task ID.")
        return

    for task in user_tasks:
        if task["id"] == task_id:
            if task["status"] == "Completed":
                print("Task is already marked as completed.")
            else:
                task["status"] = "Completed"
                print(f"Task ID {task_id} marked as completed.")
            save_tasks(tasks)
            return

    print(f"No task found with ID {task_id}.")

# --- Delete a task ---


def delete_task(username):
    tasks = load_tasks()
    user_tasks = tasks.get(username, [])

    if not user_tasks:
        print("No tasks to delete.")
        return

    print("\n--- Delete Task ---")
    try:
        task_id = int(input("Enter the task ID to delete: "))
    except ValueError:
        print("Invalid input. Please enter a numeric task ID.")
        return

    for task in user_tasks:
        if task["id"] == task_id:
            user_tasks.remove(task)
            tasks[username] = user_tasks
            save_tasks(tasks)
            print(f"Task ID {task_id} deleted.")
            return

    print(f"No task found with ID {task_id}.")

# --- Task Manager Menu ---


def task_manager_menu(username):
    while True:
        print(f"\n--- Task Manager Menu ({username}) ---")
        print("1. Add Task")
        print("2. View Tasks")
        print("3. Mark Task as Completed")
        print("4. Delete Task")
        print("5. Logout")

        choice = input("Select an option (1-5): ").strip()

        if choice == "1":
            add_task(username)
        elif choice == "2":
            view_tasks(username)
        elif choice == "3":
            mark_task_completed(username)
        elif choice == "4":
            delete_task(username)
        elif choice == "5":
            print(f"Logging out {username}...\n")
            break
        else:
            print("Invalid option. Please select 1–5.")

# --- Main Menu ---


def main():
    while True:
        print("\n=== Task Manager Authentication ===")
        print("1. Register")
        print("2. Login")
        print("3. Exit")
        choice = input("Select an option (1-3): ").strip()

        if choice == "1":
            register_user()
        elif choice == "2":
            user = login_user()
            if user:
                task_manager_menu(user)
        elif choice == "3":
            print("Exiting program.")
            break
        else:
            print("Invalid option. Please select 1, 2, or 3.")


if __name__ == "__main__":
    main()
