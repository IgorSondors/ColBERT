import pandas as pd
import os

def filter_by_category_id(df):

    # id_category = {
    #         3902: 'диктофоны, портативные рекордеры',
    #         510402: 'электронные книги',
    #         4302: 'автомобильные телевизоры, мониторы',
    #         2815: 'смарт-часы и браслеты',
    #         3901: 'портативные медиаплееры',
    #         3904: 'портативная акустика',
    #         2801: 'мобильные телефоны',
    #         3908: 'VR-гарнитуры (VR-очки, шлемы, очки виртуальной реальности, FPV очки для квадрокоптеров)',
    #         510401: 'планшетные компьютеры и мини-планшеты',
    #         2102: 'наушники, гарнитуры, наушники c микрофоном',
    #         3903: 'радиоприемники, радиобудильники, радиочасы',
    #         3907: 'магнитолы',
    #         280801: 'GPS-навигаторы'
    #         }

    id_category = {
            # 921201: 'корм для кошек',
            # 963401: 'колбасы',
            # 710502: 'стиральные машины',
            # 977135: 'ботинки, полуботинки',
            # 911906: 'парфюмерия',
            # 976132: 'туфли и лоферы',
            740101: 'кухонные мойки',
            # 7106: 'микроволновые печи',
            # 921401: 'корм для собак',
            # 963302: 'пицца'
            }
    try:
        df = df.drop(columns=['true_match', 'false_match'])
    except:
        print('true_match, false_match - колонки отсутствуют')

    valid_category_ids = set(id_category.keys())
    filtered_df = df[df['category_id'].isin(valid_category_ids)]
    return filtered_df.reset_index(drop=True)

def choose_0_5(df, df_models):
    start_index = int(input(f"Введите индекс, с которого начать выбор (от 0 до {len(df) - 1}): "))
    if start_index < 0 or start_index >= len(df):
        print(f"Некорректный индекс. Диапазон от 0 до {len(df) - 1}.")
        return df
    
    index = start_index
    while index < len(df):
        row = df.iloc[index]
        cat_id = row['category_id']
        model_gt_list = list(df_models[df_models.model_id == row['model_id']]['full_name'])
        if len(model_gt_list) >= 1:
            model_gt = model_gt_list[0]
        else:
            model_gt = 0

        top_1 = list(df_models[df_models.model_id == row['model_id_pred_1']]['full_name'])[0]
        top_2 = list(df_models[df_models.model_id == row['model_id_pred_2']]['full_name'])[0]
        top_3 = list(df_models[df_models.model_id == row['model_id_pred_3']]['full_name'])[0]
        top_4 = list(df_models[df_models.model_id == row['model_id_pred_4']]['full_name'])[0]
        top_5 = list(df_models[df_models.model_id == row['model_id_pred_5']]['full_name'])[0]

        cos_sim_1 = round(row['cosine_sims_1'], 2)
        cos_sim_2 = round(row['cosine_sims_2'], 2)
        cos_sim_3 = round(row['cosine_sims_3'], 2)
        cos_sim_4 = round(row['cosine_sims_4'], 2)
        cos_sim_5 = round(row['cosine_sims_5'], 2)

        # print(f"\ngt: {model_gt}:\n{index+1}/{len(df)}: {row['name']}")  
        print(f"{index+1}/{len(df)}:\ncategory_id: {cat_id}\nmodel: {model_gt}\noffer: {row['name']}") 
        # print(f"1) {top_1}\n2) {top_2}\n3) {top_3}\n4) {top_4}\n5) {top_5}\n0) Нет правильного")
        print(f"1) {top_1} --> {cos_sim_1}\n2) {top_2} --> {cos_sim_2}\n3) {top_3} --> {cos_sim_3}\n4) {top_4} --> {cos_sim_4}\n5) {top_5} --> {cos_sim_5}\n0) Нет правильного")
        manual_input = input("Введите число от 0 до 5, '-' для перемещения назад, '+' для перемещения вперед, 'q' для выхода: ")
        
        # Проверка введенного значения и вставка соответствующего значения в колонку "model_id_manual"
        if manual_input == 'q':
            break  # Выход из цикла
        elif manual_input.isdigit():
            manual_input = int(manual_input)
            if manual_input == 0:
                df.at[index, 'model_id_manual'] = 0
                index += 1
                # Очистка вывода в терминале
                os.system('cls' if os.name == 'nt' else 'clear')
            elif manual_input >= 1 and manual_input <= 5:
                df.at[index, 'model_id_manual'] = row[f'model_id_pred_{manual_input}']
                index += 1 
                # Очистка вывода в терминале
                os.system('cls' if os.name == 'nt' else 'clear')
            else:
                print(f"{manual_input} --> Некорректный ввод. Введите число от 0 до 5.")
        elif manual_input == '-': # Перемещение к предыдущей записи
            if index > 0:
                index -= 1
        elif manual_input == '+': # Перемещение к следующей записи
            index += 1 
        else:
            print(f"{manual_input} --> Некорректный ввод.")
    return df

def yes_no(df, df_models):
    start_index = int(input(f"Введите индекс, с которого начать выбор (от 0 до {len(df) - 1}): "))
    if start_index < 0 or start_index >= len(df):
        print(f"Некорректный индекс. Диапазон от 0 до {len(df) - 1}.")
        return df
    
    index = start_index
    while index < len(df):
        row = df.iloc[index]
        cat_id = row['category_id']
        model_gt_list = list(df_models[df_models.model_id == row['model_id']]['full_name'])
        if len(model_gt_list) >= 1:
            model_gt = model_gt_list[0]
        else:
            model_gt = 0
            
        print(f"{index+1}/{len(df)}:\ncategory_id: {cat_id}\nmodel: {model_gt}\noffer: {row['name']}")  
        print("1) да\n2) нет\n0) хз")

        manual_input = input("Введите число от 0 до 2, '-' для перемещения назад, '+' для перемещения вперед, 'q' для выхода: ")
        
        # Проверка введенного значения и вставка соответствующего значения в колонку "model_id_correct"
        if manual_input == 'q':
            break  # Выход из цикла
        elif manual_input.isdigit():
            manual_input = int(manual_input)
            if manual_input == 0:
                df.at[index, 'model_id_correct'] = 0
                index += 1
                # Очистка вывода в терминале
                os.system('cls' if os.name == 'nt' else 'clear')
            elif manual_input >= 1 and manual_input <= 2:
                df.at[index, 'model_id_correct'] = manual_input
                index += 1 
                # Очистка вывода в терминале
                os.system('cls' if os.name == 'nt' else 'clear')
            else:
                print(f"{manual_input} --> Некорректный ввод. Введите число от 0 до 2.")
        elif manual_input == '-': # Перемещение к предыдущей записи
            if index > 0:
                index -= 1
        elif manual_input == '+': # Перемещение к следующей записи
            index += 1 
        else:
            print(f"{manual_input} --> Некорректный ввод.")
    return df

if __name__=='__main__':
    

    pth_models = "/home/sondors/Documents/price/BERT_data/data/10-04-2024_Timofey/2801_models_Apple.csv"
    pth_src = "/home/sondors/Documents/price/BERT_data/data/17-04-2024_Timofey/MODEL_ID=0/2801_Apple_offers_active_model_id0_before_10-04-2024_cos_no_repeats.csv"
    pth_dst = "/home/sondors/Documents/price/BERT_data/data/17-04-2024_Timofey/MODEL_ID=0/2801_Apple_offers_active_model_id0_before_10-04-2024_cos_no_repeats.csv"

    df_models = pd.read_csv(pth_models, sep=";")
    df_offers = pd.read_csv(pth_dst, sep=";")

    # df = yes_no(df_offers, df_models)
    df = choose_0_5(df_offers, df_models)
    df.to_csv(pth_dst, sep=';', index=False)

    print(df)