# Import the necessary libraries
import random
import time

# Define a list of words
words = ['cat', 'dog', 'fish', 'bird', 'rabbit']

# Shuffle the list of words
random.shuffle(words)

# Display each word in the shuffled list and ask the user to type it in
for word in words:
    print(word)
    user_input = input('Please type the word you see above: ')
    
    # Check if the user's input matches the displayed word
    if user_input == word:
        print('Correct!')
    else:
        print('Incorrect.')
    
    # Pause for 1 second before displaying the next word
    time.sleep(1)
