#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pickle
class person_info:
    def __init__(self):
        self.person_db=[]
        self.secret_dates = ['01/05/2010', '05/02/2013']
        
    def get_input(self):
        
        name=input("\nEnter the person name:")
        dob=input("Enter the dob in dd/mm/yyyy:")
        self.person_db.append((name,dob))
        self.save_file()
        
    def display_dob(self):
        name=input("\nEnter the person's name:")
        for person in self.person_db:
            if person[0] == name:
                if person[1] in self.secret_dates:
                    print("Secret, Try another person")
                else:
                    print("Date of Birth:", person[1])
                return
        print("\nPerson name not found.")
        
    def save_file(self):
            with open("c:\\Users\\epvin\\problem1_data_file.pickle", 'wb') as file:
                 pickle.dump(self.person_db, file)
            print("Data saved successfully.")
            print("\nContents of self.person_db:\n", self.person_db)
        
        
    def load_file(self):
            with open("c:\\Users\\epvin\\problem1_data_file.pickle", "rb") as file:
                self.person_db = pickle.load(file)
            print("Data loaded successfully.")
        
    
def main():
    person_file = person_info()
    person_file.load_file()
    while True:
        print("\n-----Managing Personal Info______")
        print("\n1. Adding a Person's name and DOB")
        print("2. Displaying the Date of Birth for a particular person")
        print("3. Exit")

        choice = input("\nEnter your choice: ")

        if choice == "1":
            person_file.get_input()
        elif choice == "2":
            person_file.display_dob()
        elif choice == "3":
            print("....Exiting...")
            
            break
        else:
            print("Pick a choice from above. Please try again.")
            
main()



# In[ ]:




