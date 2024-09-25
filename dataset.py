import json
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

def datatable(data, chart_type):
    data_table = ''
    num = len(data)
    
    if len(data)  == 2:
        temp = []
        temp.append(f"대상: {data[0][4]}")
        temp.append(f"제목: {data[0][0]}")
        temp.append(f"유형: {' '.join(chart_type[0:2])}")
        temp.append(f"{data[0][5]} | {data[0][1][0]}({data[0][3]}) | {data[1][1][0]}({data[1][3]})")
    
        x_axis = data[0][7]
        for idx, x in enumerate(x_axis):
            temp.append(f"{x} | {data[0][2][0][idx]} | {data[1][2][0][idx]}")
            
        data_table = '\n'.join(temp)
    else:
        for n in range(num):
            temp = []
            
            title, legend, datalabel, unit, base, x_axis_title, y_axis_title, x_axis, y_axis = data[n]
            legend = [element + f"({unit})" for element in legend]
            
            if len(legend) > 1:
                temp.append(f"대상: {base}")
                temp.append(f"제목: {title}")
                temp.append(f"유형: {' '.join(chart_type[0:2])}")
                temp.append(f"{x_axis_title} | {' | '.join(legend)}")
                
                if chart_type[2] == "원형":
                    datalabel = sum(datalabel, [])
                    temp.append(f"{' | '.join([str(d) for d in datalabel])}")
                    data_table = '\n'.join(temp)
                else:
                    axis = y_axis if chart_type[2] == "가로 막대형" else x_axis
                    for idx, (x, d) in enumerate(zip(axis, datalabel)):
                        temp_d = [str(e) for e in d]
                        temp_d = " | ".join(temp_d)
                        row = f"{x} | {temp_d}"
                        temp.append(row)
                    data_table = '\n'.join(temp)
            
            else:
                temp.append(f"대상: {base}")
                temp.append(f"제목: {title}")
                temp.append(f"유형: {' '.join(chart_type[0:2])}")
                temp.append(f"{x_axis_title} | {unit}")
                
                axis = y_axis if chart_type[2] == "가로 막대형" else x_axis
                datalabel = datalabel[0]
                
                for idx, x in enumerate(axis):
                    row = f"{x} | {str(datalabel[idx])}"
                    temp.append(row)
                data_table = '\n'.join(temp)
    
    return data_table
    

def chart_data(data):
    datatable = []    
    num = len(data)
    for n in range(num):
        title = data[n]['title'] if data[n]['is_title'] else ''
        legend = data[n]['legend'] if data[n]['is_legend'] else ''
        datalabel = data[n]['data_label'] if data[n]['is_datalabel'] else [0]
        unit = data[n]['unit'] if data[n]['is_unit'] else ''
        base = data[n]['base'] if data[n]['is_base'] else ''
        x_axis_title = data[n]['axis_title']['x_axis']
        y_axis_title = data[n]['axis_title']['y_axis']
        x_axis = data[n]['axis_label']['x_axis'] if data[n]['is_axis_label_x_axis'] else [0]
        y_axis = data[n]['axis_label']['y_axis'] if data[n]['is_axis_label_y_axis'] else [0]
        
        if len(legend) > 1:
            datalabel = np.array(datalabel).transpose().tolist()          
                
        datatable.append([title, legend, datalabel, unit, base, x_axis_title, y_axis_title, x_axis, y_axis])
        
    return datatable

def main(IMAGE_PATH, JSON_PATH, SAVE_PATH):
    image_file_list = os.listdir(IMAGE_PATH)
    
    dataset = []
    for image_file in tqdm(image_file_list):
        data_id = image_file.split(".jpg")[0]
        json_file_name = image_file.replace("Source", "Label").split(".jpg")[0] + ".json"
        
        with open(JSON_PATH + json_file_name, "r") as f:
            json_data = json.load(f)
        
        chart_multi = json_data['metadata']['chart_multi']
        chart_main = json_data['metadata']['chart_main']
        chart_sub = json_data['metadata']['chart_sub']
        chart_type = [chart_multi, chart_sub, chart_main]
        
        chart_annotations = json_data['annotations']
        
        description = "설명: " + json_data["description"].strip()
        summary = "요약: " + "\\".join(json_data["summary"])
        texts = description + "\n" + summary
        
        try:
            charData = chart_data(chart_annotations)
        except:
            pass
        
        try:
            dataTable = datatable(charData, chart_type)
            dataset.append([image_file, dataTable, texts, data_id])
        except:
            pass
            
    df = pd.DataFrame(dataset, columns=["file_name", "text", "description", "data_id"])
    
    if not os.path.isdir(SAVE_PATH):
        os.mkdir(SAVE_PATH)
    df.to_csv(SAVE_PATH + "metadata.csv", index=False, header=True)
    print("save")
    
    return df