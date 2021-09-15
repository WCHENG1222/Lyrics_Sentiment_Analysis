from opencc import OpenCC

def main():
    with open(r'schi.txt', 'r', encoding='utf-8') as opentxt:
        with open(r'tchi.txt', 'w', encoding='utf8') as writetxt:
            for line in opentxt.readlines():
                terms = line.strip()
                # 簡繁轉換 Simplified Chinese to Traditional Chinese (Taiwan standard, with phrases)
                openCC = OpenCC('s2twp')
                terms_tw_zh = openCC.convert(terms)
                writetxt.write(terms_tw_zh + "\n")
            writetxt.close()
        opentxt.close()


if __name__ == '__main__':
    main()