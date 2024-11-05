text = '''

'''
process_ids = []
lines = text.strip().split("\n")
for line in lines:
    columns = line.split()
    if len(columns) >= 2:
        process_ids.append(columns[1])
print("kill -9 "+" ".join(process_ids))
