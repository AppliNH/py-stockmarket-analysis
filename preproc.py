import pandas as pd

def social_stocks_pre_processing():
    socials_df = pd.read_csv('social_medias_stock_df.csv')
    print(socials_df)


def automobile_stocks_pre_processing(df):
    df_without_GM = df.drop(columns=['GM']).copy()
    df_without_GM.dropna(inplace=True)
    df_without_GM.reset_index(inplace=True)
    # print(df_without_GM)

    df_GM = df.drop(columns=['PUGOY', "BMWYY", "VLKAF"]).copy()
    df_GM.dropna(inplace=True)
    df_GM.reset_index(inplace=True)
    df_GM.drop(columns="t", inplace=True)
    df_GM["t"] = df_without_GM["t"]
    # print(df_GM)


    autos_df_final = pd.merge(df_GM, df_without_GM)
    autos_df_final.set_index("t", inplace=True)
    # print(autos_df_final)
    return autos_df_final
    

# def automobile_stocks_pre_processing():
#     autos_df = pd.read_csv('automobile_stock_df.csv', index_col=0)
#     print(autos_df)


#     autos_df_without_GM = autos_df.drop(columns=['GM', 'GM.1', 'GM.2', 'GM.3', 'GM.4']).copy()
#     autos_df_without_GM.reset_index(inplace=True)

#     autos_df_without_GM.dropna(inplace=True)


#     time = autos_df_without_GM["index"].copy()
#     time = time.reset_index()
#     time.drop(columns="level_0", inplace=True)
#     autos_df_without_GM.drop(columns="index", inplace=True)
#     autos_df_without_GM.reset_index(inplace=True)
#     autos_df_without_GM.drop(columns="index", inplace=True)

#     autos_df_without_GM["t"] = time

#     # print(autos_df_without_GM)
#     # print(time)


#     GM = autos_df[['GM', 'GM.1', 'GM.2', 'GM.3', 'GM.4']].copy()
#     GM = GM.iloc[1:]
#     GM.reset_index(inplace=True)
#     GM.drop(columns="index", inplace=True)
#     GM.dropna(inplace=True)
#     GM.reset_index(inplace=True)
#     GM.drop(columns="index", inplace=True)

#     GM["t"]=time

#     # print(GM)

#     autos_df_final = pd.merge(GM,autos_df_without_GM)

#     rename_map = {
#         "GM": "GM|c",
#         "GM.1":"GM|h",
#         "GM.2":"GM|l",
#         "GM.3":"GM|o",
#         "GM.4":"GM|v",
#         "BMWYY": "BMWYY|c",
#         "BMWYY.1":"BMWYY|h",
#         "BMWYY.2":"BMWYY|l",
#         "BMWYY.3":"BMWYY|o",
#         "BMWYY.4":"BMWYY|v",
#         "VLKAF": "VLKAF|c",
#         "VLKAF.1":"VLKAF|h",
#         "VLKAF.2":"VLKAF|l",
#         "VLKAF.3":"VLKAF|o",
#         "VLKAF.4":"VLKAF|v",
#         "PUGOY": "PUGOY|c",
#         "PUGOY.1":"PUGOY|h",
#         "PUGOY.2":"PUGOY|l",
#         "PUGOY.3":"PUGOY|o",
#         "PUGOY.4":"PUGOY|v"
#     }

#     autos_df_final.rename(columns=rename_map, inplace=True)
#     autos_df_final.set_index("t", inplace=True)
#     print(autos_df_final.head(10))
