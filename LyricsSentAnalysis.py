# Taiwan Music Lyrics Sentiment Analysis

import jieba
import csv


def main():
    countAnalysis = 0

    # 存取文本所有不同的情緒詞彙與副詞
    termList_Happy = []
    termList_Great = []
    termList_Anger = []
    termList_Sadness = []
    termList_Fear = []
    termList_Wicked = []
    termList_Surprise = []
    termList_Not = []
    termList_Degree = []
    termList_Freq = []

    # 載入結巴預設字典與使用者自定義字典
    # 預設字典 - 取用自: fukuball https://github.com/fukuball/Head-first-Chinese-text-segmentation/blob/master/data/dict.txt.big
    print('預設字典讀取中...')
    jieba.set_dictionary(r'NLP_senAnalysis\Dictionary\dict.txt.big_fukuball.txt')
    print('使用者自定義字典讀取中...')
    jieba.load_userdict(r'NLP_senAnalysis\Dictionary\dict_addword.txt')
    print('停用詞字典讀取中...')
    stopword = [w.strip() for w in open(r'NLP_senAnalysis\Dictionary\dict_stopword.txt',
                                        'r', encoding='utf-8').readlines()]

    # 情緒詞 - 取用自: 中國大連理工大學信息檢索研究室之中文情感詞彙本體 http://ir.dlut.edu.cn/info/1013/1142.htm
    sentiment_dict = []
    print('情緒詞典讀取中...')
    with open(r'NLP_senAnalysis\Dictionary\dict_ChiAffectLexiOnto.csv',
              'r', encoding='utf-8', newline='') as csvfile_sentDict:
        reader = csv.DictReader(csvfile_sentDict, delimiter=',')
        for row in reader:
            sentiment_dictTemp = [
                {'term': row['term'], 'weight': row['weight'], 'senClass': row['senClass'], 'senTag': row['senTag']}
            ]
            sentiment_dict.append(sentiment_dictTemp)
        csvfile_sentDict.close()
        print('情緒詞典共 %d 個詞' % (len(sentiment_dict)))

    # 否定詞 - wcheng1222 自建
    print('否定詞典讀取中...')
    notword_dict = [w.strip() for w in open(r'NLP_senAnalysis\Dictionary\dict_notword.txt',
                                            'r', encoding='utf-8').readlines()]
    print('否定詞典共 %d 個詞' % (len(notword_dict)))

    # 程度詞與頻率詞 - 整理自:
    #               中央研究院中文詞知識庫小組之廣義知網知識本體架構2.0版(線上瀏覽) http://ehownet.iis.sinica.edu.tw/ehownet.php
    #             & 知網-中文信息結構庫之情感分析用詞語集(beta版) http://www.keenage.com/download/sentiment.rar
    degfreqword_dict = []
    print('程度頻率詞典讀取中...')
    with open(r'NLP_senAnalysis\Dictionary\dict_degfreqword.csv',
              'r', encoding='utf-8', newline='') as csvfile_degfreqDict:
        reader = csv.DictReader(csvfile_degfreqDict, delimiter=',')
        for row in reader:
            degfreqword_dictTemp = [
                {'term': row['term'], 'weight': row['weight'], 'advClass': row['advClass']}
            ]
            degfreqword_dict.append(degfreqword_dictTemp)
        csvfile_degfreqDict.close()
        print('程度頻率詞典共 %d 個詞' % (len(degfreqword_dict)))

    print("### Taiwan music lyrics sentiment analysis project ###")
    analyFilename = input('請輸入情感分析計算的檔案名稱：')

    # csv 開啟寫入: 文本所有情緒分析值
    with open('情緒分析值_' + analyFilename, 'a', newline='', encoding='utf8') as csvfile_sentAnalysis_w:
        writer = csv.writer(csvfile_sentAnalysis_w, delimiter=',')
        writer.writerow(['ID', '歌名', '創作人', '分類', '發佈時間', '情緒詞彙數', '斷詞數',
                         '樂', '好', '怒', '哀', '懼', '惡', '驚', '正面情緒值', '負面情緒值', '正負面情緒值',
                         '詞均樂', '詞均好', '詞均怒', '詞均哀', '詞均懼', '詞均惡', '詞均驚',
                         '詞均正面情緒值', '詞均負面情緒值', '詞均正負面情緒值'])

        # 讀取欲分析的檔案
        with open(analyFilename, 'r', encoding='utf-8') as csvfile_rawdata:
            reader = csv.DictReader(csvfile_rawdata, delimiter=',')

            for row_rawdata in reader:
                seg_lyrics = jieba.cut(row_rawdata['歌詞'], cut_all=False)
                # 停用詞剔除
                segList = [w for w in seg_lyrics if w not in stopword]

                # 取出分析目標之斷詞列表含有: 情緒、否定、程度、頻率之詞彙，存取至列表中
                seginSentList = []
                # 斷詞列表攤平
                for index, word in enumerate(segList):
                    # 情緒字典列表攤平
                    for rows_sent in sentiment_dict:
                        for row_sent in rows_sent:
                            # 判斷斷詞是否為情緒詞
                            if word == row_sent['term']:
                                seginSentList.append(
                                    [row_sent['term'], row_sent['weight'], row_sent['senClass'], row_sent['senTag']])
                    # 程度與頻率字典列表攤平
                    for rows_degfreq in degfreqword_dict:
                        for row_degfreq in rows_degfreq:
                            # 判斷斷詞是否為程度或頻率詞
                            if word == row_degfreq['term']:
                                seginSentList.append(
                                    [row_degfreq['term'], row_degfreq['weight'], 'sentweight', row_degfreq['advClass']])
                    # 否定列表攤平
                    for index_n, word_n in enumerate(notword_dict):
                        # 判斷斷詞是否為否定詞，賦值為 -1 表示相反
                        if word == word_n:
                            seginSentList.append(
                                [word_n, -1, 'sentweight', '否定詞'])

                # 各個情緒值(樂、好、怒、哀、懼、惡、驚)與權重(程度、頻率)之初始化
                scoreHappy = 0
                scoreGreat = 0
                scoreAnger = 0
                scoreSadness = 0
                scoreFear = 0
                scoreWicked = 0
                scoreSurprise = 0
                weight = 1

                # 情緒值計算
                # 定義一個情緒詞組為: ...情緒詞1...[否定詞...程度/頻率詞...情緒詞2]...
                # ，即由兩個情緒詞彙之間所有否定詞與程度/頻率詞以及後一情緒詞(情緒詞2)所構成
                # 情緒詞組之情緒值 = 否定詞 * 程度/頻率詞 * 情緒詞
                for matrix in seginSentList:
                    # 若詞彙為否定詞、程度詞或頻率詞，作為情緒詞之權重值
                    if matrix[2] == 'sentweight':
                        weight *= float(matrix[1])
                    # 若詞彙為情緒詞，依照不同情緒: 樂、好、怒、哀、懼、惡、驚 計算
                    else:
                        if matrix[2] == '樂':
                            scoreHappy += (float(matrix[1]) * weight)
                            weight = 1
                        elif matrix[2] == '好':
                            scoreGreat += (float(matrix[1]) * weight)
                            weight = 1
                        elif matrix[2] == '怒':
                            scoreAnger += (float(matrix[1]) * weight)
                            weight = 1
                        elif matrix[2] == '哀':
                            scoreSadness += (float(matrix[1]) * weight)
                            weight = 1
                        elif matrix[2] == '懼':
                            scoreFear += (float(matrix[1]) * weight)
                            weight = 1
                        elif matrix[2] == '惡':
                            scoreWicked += (float(matrix[1]) * weight)
                            weight = 1
                        elif matrix[2] == '驚':
                            scoreSurprise += (float(matrix[1]) * weight)
                            weight = 1

                # 正面情緒值 = 樂 + 好 + 驚
                scorePositive = scoreHappy + scoreGreat + scoreSurprise
                # 負面情緒值 = 怒 + 哀 + 懼 + 惡
                scoreNegative = scoreAnger + scoreSadness + scoreFear + scoreWicked
                # 正負面情緒值 = 正面情緒值 - 負面情緒值
                scoreSentiment = (scorePositive - scoreNegative)

                # 詞彙存取
                for matrix in seginSentList:
                    if matrix[2] == 'sentweight':
                        if matrix[3] == '否定詞':
                            termList_Not.append(matrix[0])
                        elif matrix[3] == '極其' or matrix[3] == '超' or matrix[3] == '很' or matrix[3] == '較' or matrix[
                            3] == '稍' or \
                                matrix[3] == '不足':
                            termList_Degree.append(matrix[0])
                        elif matrix[3] == '總是' or matrix[3] == '經常' or matrix[3] == '偶爾':
                            termList_Freq.append(matrix[0])
                    elif matrix[2] == '樂':
                        termList_Happy.append(matrix[0])
                    elif matrix[2] == '好':
                        termList_Great.append(matrix[0])
                    elif matrix[2] == '怒':
                        termList_Anger.append(matrix[0])
                    elif matrix[2] == '哀':
                        termList_Sadness.append(matrix[0])
                    elif matrix[2] == '懼':
                        termList_Fear.append(matrix[0])
                    elif matrix[2] == '惡':
                        termList_Wicked.append(matrix[0])
                    elif matrix[2] == '驚':
                        termList_Surprise.append(matrix[0])

                avg_scoreHappy = scoreHappy / len(segList)
                avg_scoreGreat = scoreGreat / len(segList)
                avg_scoreAnger = scoreAnger / len(segList)
                avg_scoreSadness = scoreSadness / len(segList)
                avg_scoreFear = scoreFear / len(segList)
                avg_scoreWicked = scoreWicked / len(segList)
                avg_scoreSurprise = scoreSurprise / len(segList)
                avg_scorePositive = scorePositive / len(segList)
                avg_scoreNegative = scoreNegative / len(segList)
                avg_scoreSentiment = scoreSentiment / len(segList)

                # 單首歌曲情緒值結果輸出
                print(seginSentList)
                print(len(segList))
                print('情緒值： 樂：%f 好：%f 怒：%f 哀：%f 懼：%f 惡：%f 驚：%f 正面情緒值：%f 負面情緒值：%f 正負面情緒值：%f' % (
                    scoreHappy, scoreGreat, scoreAnger, scoreSadness, scoreFear, scoreWicked, scoreSurprise,
                    scorePositive, scoreNegative, scoreSentiment))
                print('詞均情緒值： 樂：%f 好：%f 怒：%f 哀：%f 懼：%f 惡：%f 驚：%f 正面情緒值：%f 負面情緒值：%f 正負面情緒值：%f' % (
                    avg_scoreHappy, avg_scoreGreat, avg_scoreAnger, avg_scoreSadness, avg_scoreFear,
                    avg_scoreWicked, avg_scoreSurprise, avg_scorePositive, avg_scoreNegative,
                    avg_scoreSentiment))
                countAnalysis += 1
                print(row_rawdata['ID'], row_rawdata['歌名'], row_rawdata['創作人'])
                print('第 %d 首歌分析完成！！' % countAnalysis)

                # csv 寫入: 文本所有情緒分析值
                writer.writerow([row_rawdata['ID'], row_rawdata['歌名'], row_rawdata['創作人'], row_rawdata['分類'],
                                 row_rawdata['發佈時間'], len(seginSentList), len(segList),
                                 scoreHappy, scoreGreat, scoreAnger, scoreSadness, scoreFear,
                                 scoreWicked, scoreSurprise, scorePositive, scoreNegative, scoreSentiment,
                                 avg_scoreHappy, avg_scoreGreat, avg_scoreAnger, avg_scoreSadness, avg_scoreFear,
                                 avg_scoreWicked, avg_scoreSurprise, avg_scorePositive, avg_scoreNegative,
                                 avg_scoreSentiment])

    # csv 開啟寫入: 文本所有不同的情緒詞彙與副詞次數計算
    with open('情緒詞彙與副詞列表_' + analyFilename, 'a', newline='', encoding='utf8') as csvfile_termlist_w:
        writer = csv.writer(csvfile_termlist_w, delimiter=',')
        writer.writerow(['樂term', '樂count', '好term', '好count', '怒term', '怒count',
                         '哀term', '哀count', '懼term', '懼count', '惡term', '惡count',
                         '驚term', '驚count', '否定詞', '否定詞count', '程度詞', '程度詞count', '頻率詞', '頻率詞count'])
        for term_Happy in termList_Happy:
            writer.writerow([term_Happy, termList_Happy.count(term_Happy)])
        for term_Great in termList_Great:
            writer.writerow([None, None,
                             term_Great, termList_Great.count(term_Great)])
        for term_Anger in termList_Anger:
            writer.writerow([None, None, None, None,
                             term_Anger, termList_Anger.count(term_Anger)])
        for term_Sadness in termList_Sadness:
            writer.writerow([None, None, None, None, None, None,
                             term_Sadness, termList_Sadness.count(term_Sadness)])
        for term_Fear in termList_Fear:
            writer.writerow([None, None, None, None, None, None, None, None,
                             term_Fear, termList_Fear.count(term_Fear)])
        for term_Wicked in termList_Wicked:
            writer.writerow([None, None, None, None, None, None, None, None, None, None,
                             term_Wicked, termList_Wicked.count(term_Wicked)])
        for term_Surprise in termList_Surprise:
            writer.writerow([None, None, None, None, None, None, None, None, None, None, None, None,
                             term_Surprise, termList_Surprise.count(term_Surprise)])
        for term_Not in termList_Not:
            writer.writerow([None, None, None, None, None, None, None, None, None, None, None, None, None, None,
                             term_Not, termList_Not.count(term_Not)])
        for term_Degree in termList_Degree:
            writer.writerow([None, None, None, None, None, None, None, None, None, None, None, None, None, None,
                             None, None, term_Degree, termList_Degree.count(term_Degree)])
        for term_Freq in termList_Freq:
            writer.writerow([None, None, None, None, None, None, None, None, None, None, None, None, None, None,
                             None, None, None, None, term_Freq, termList_Freq.count(term_Freq)])
        csvfile_termlist_w.close()
        csvfile_rawdata.close()
    csvfile_sentAnalysis_w


if __name__ == '__main__':
    main()
