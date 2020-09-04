class TextTransform:
    def __init__(self):
        char_map_str = """
        <BLANK> 0
        ' 1
        <SPACE> 2
        a 3
        b 4
        c 5
        d 6
        e 7
        f 8
        g 9
        h 10
        i 11
        j 12
        k 13
        l 14
        m 15
        n 16
        o 17
        p 18
        q 19
        r 20
        s 21
        t 22
        u 23
        v 24
        w 25
        x 26
        y 27
        z 28
        """
        self.char_map = {}
        self.index_map = {}
        for line in char_map_str.strip().split("\n"):
            ch, index = line.split()
            self.char_map[ch] = int(index)
            self.index_map[int(index)] = ch
        self.index_map[2] = " "

    def text_to_int(self, text):
        """ Use a character map and convert text to an integer sequence """
        int_sequence = []
        for c in text:
            if c == " ":
                ch = self.char_map["<SPACE>"]
            else:
                ch = self.char_map[c]
            int_sequence.append(ch)
        return int_sequence

    def int_to_text(self, labels):
        """ Use a character map and convert integer labels to an text sequence """
        string = []
        for i in labels:
            string.append(self.index_map[i])
        return "".join(string)
