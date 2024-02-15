import pandas as pd
import os

def filter_by_category_id(df, id_category):
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
        model_gt = list(df_models[df_models.model_id == row['model_id']]['full_name'])[0]
        top_1 = list(df_models[df_models.model_id == row['model_id_pred_1']]['full_name'])[0]
        top_2 = list(df_models[df_models.model_id == row['model_id_pred_2']]['full_name'])[0]
        top_3 = list(df_models[df_models.model_id == row['model_id_pred_3']]['full_name'])[0]
        top_4 = list(df_models[df_models.model_id == row['model_id_pred_4']]['full_name'])[0]
        top_5 = list(df_models[df_models.model_id == row['model_id_pred_5']]['full_name'])[0]
        print(f"\ngt: {model_gt}:\n{index+1}/{len(df)}: {row['name']}")  
        print(f"1) {top_1}\n2) {top_2}\n3) {top_3}\n4) {top_4}\n5) {top_5}\n0) Нет правильного")

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
        model_gt = list(df_models[df_models.model_id == row['model_id']]['full_name'])[0]
        print(f"{index+1}/{len(df)}:\noffer: {row['name']}\nmodel: {model_gt}")  
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

    id_category = {
            3902: 'диктофоны, портативные рекордеры',
            510402: 'электронные книги',
            4302: 'автомобильные телевизоры, мониторы',
            2815: 'смарт-часы и браслеты',
            3901: 'портативные медиаплееры',
            3904: 'портативная акустика',
            2801: 'мобильные телефоны',
            3908: 'VR-гарнитуры (VR-очки, шлемы, очки виртуальной реальности, FPV очки для квадрокоптеров)',
            510401: 'планшетные компьютеры и мини-планшеты',
            2102: 'наушники, гарнитуры, наушники c микрофоном',
            3903: 'радиоприемники, радиобудильники, радиочасы',
            3907: 'магнитолы',
            280801: 'GPS-навигаторы'
            }

    pth_models = "/home/sondors/Documents/price/ColBERT_data/18_categories/test/models_18_categories.csv"
    pth_src = "/home/sondors/Documents/price/ColBERT/proverka/colbert-5387_offers_top_n_exclude_top5.csv"
    pth_dst = "/home/sondors/Documents/price/ColBERT/proverka/colbert-5387_offers_top_n_exclude_top5_manual.csv"

    df_models = pd.read_csv(pth_models, sep=";")
    df_offers = pd.read_csv(pth_src, sep=";")
    # df_offers = filter_by_category_id(df_offers, id_category)

    df = choose_0_5(df_offers[:10], df_models)
    df.to_csv(pth_dst, sep=';', index=False)

    print(df)