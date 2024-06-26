import random

with open("wolframall_bad.txt", 'r') as file:
    lines1 = ["\"" + line.strip() + "\",0" for line in file.readlines()]
with open("wolframall_good.txt", 'r') as file:
    lines2 = ["\"" + line.strip() + "\",1" for line in file.readlines()]

lines = lines1 + lines2
# Shuffle the lines randomly
random.shuffle(lines)

# Write shuffled lines back to the file
with open("wolframall_mix.csv", 'w') as file:
    file.writelines(line + '\n' for line in lines)


print("Lines shuffled successfully.")

