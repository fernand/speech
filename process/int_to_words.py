singles = ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
tens = [
    "",
    "",
    "twenty",
    "thirty",
    "forty",
    "fifty",
    "sixty",
    "seventy",
    "eighty",
    "ninety",
]
teens = [
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
]
suffix = ["", "thousand", "million", "billion", "trillion", "quadrillion", "zillion"]


def name_chunk(chunk, add_and=False):
    # container for the evaluated digits
    name_list = []
    # have to check for 0 on this digit, seeing as "" hundred will make no sense
    if not int(chunk[0]) == 0:
        name_list.append(singles[int(chunk[0])])
        name_list.append("hundred")
    # check if in teens before moving on
    if int(chunk[1]) == 1:
        if add_and:
            name_list.append("and")
        name_list.append(teens[int(chunk[2])])
    else:
        # have to check for 0 here due to same reason as before
        if add_and and not int(chunk[1:]) == 0:
            name_list.append("and")
        name_list.append(tens[int(chunk[1])])
        name_list.append(singles[int(chunk[2])])
    return name_list


def name_number(number):
    # The simplest case
    if number == 0:
        return "zero"

    # container for the evaluated digits
    name_list = []

    # add negative?
    if number < 0:
        name_list.append("negative")
        number = -number

    # Pad such that it can be broken into threes
    number = str(number)
    while not len(number) % 3 == 0:
        number = "0" + number

    # break number into chunks
    sections = [number[ii : ii + 3] for ii in range(0, len(number), 3)]

    # name the chunks, and add a suffix
    for ii in range(len(sections)):
        # if the section is zero, it can be skipped
        if not int(sections[ii]) == 0:
            # if it is the last chunk, you have to add "and"
            name_list.extend(name_chunk(sections[ii], ii + 1 == len(sections)))
            # limited by the ammount of suffix's defined
            name_list.append(suffix[len(sections) - (ii + 1)])

    if len(name_list) > 0 and name_list[0] == "and":
        name_list = name_list[1:]

    # remove undefined numbers
    while "" in name_list:
        name_list.remove("")

    # return that stuff
    return " ".join(name_list)
