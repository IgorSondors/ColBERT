import pandas as pd

def filter_by_category_id(df, id_category):
    try:
        df = df.drop(columns=['true_match', 'false_match'])
    except:
        print('true_match, false_match - колонки отсутствуют')

    valid_category_ids = set(id_category.keys())
    filtered_df = df[df['category_id'].isin(valid_category_ids)]
    return filtered_df.reset_index(drop=True)

def choose_0_5(df, df_models):
    for index, row in df.iterrows():
        model_gt = list(df_models[df_models.model_id == row['model_id']]['full_name'])[0]
        top_1 = list(df_models[df_models.model_id == row['model_id_pred_1']]['full_name'])[0]
        top_2 = list(df_models[df_models.model_id == row['model_id_pred_2']]['full_name'])[0]
        top_3 = list(df_models[df_models.model_id == row['model_id_pred_3']]['full_name'])[0]
        top_4 = list(df_models[df_models.model_id == row['model_id_pred_4']]['full_name'])[0]
        top_5 = list(df_models[df_models.model_id == row['model_id_pred_5']]['full_name'])[0]
        print(f"\n{model_gt}:\n{row['name']}")  
        print(f"1) {top_1}\n2) {top_2}\n3) {top_3}\n4) {top_4}\n5) {top_5}\n")

        # Запрос ввода от пользователя
        manual_input = input("Введите число от 0 до 5: ")
        
        # Проверка введенного значения и вставка соответствующего значения в колонку "manual"
        if manual_input.isdigit():
            manual_input = int(manual_input)
            if manual_input == 0:
                df.at[index, 'model_id_manual'] = 0
            elif manual_input >= 1 and manual_input <= 5:
                df.at[index, 'model_id_manual'] = row[f'model_id_pred_{manual_input}']
            else:
                print("Некорректный ввод. Введите число от 0 до 5.")
        else:
            print("Некорректный ввод. Введите число от 0 до 5.")
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