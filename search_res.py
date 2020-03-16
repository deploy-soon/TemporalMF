import os
import json


def get_res_list(data_name):
    file_list = os.listdir(os.path.join("res", data_name))
    return file_list

def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as fin:
        return json.load(fin)

def load_res(data_name):
    res = []
    for file_name in get_res_list(data_name):
        res_json = load_json(os.path.join("res", data_name, file_name))
        res.append(res_json)
    return res

def check(row, col, args, res):
    def _check_dict(condition, total):
        for key, value in condition.items():
            if key not in total:
                return False
            if total[key] != value:
                return False
        return True

    res = [r for r in res if _check_dict(r, row)]
    res = [r for r in res if _check_dict(r, col)]
    res = [r for r in res if _check_dict(r, args)]
    return res[0] if res else {}

def print_res(rows, cols, data_name, base_args, sep="\t"):

    def print_item(item):
        str_list =  ["{}={}".format(key, value) for key, value in item.items()]
        return " ".join(str_list)

    def print_header(info, sep="\t"):
        _opt = "table"
        for i in info:
            _opt += sep
            _opt += print_item(i)
        return _opt

    opt = print_header(cols)
    opt += "\n"
    res = load_res(data_name)
    for row in rows:
        opt += print_item(row)
        opt += sep
        for col in cols:
            item = check(row, col, base_args, res)
            vali_loss = item.get("vali_loss", 0.0)
            opt += "{:.4}".format(vali_loss)
            opt += sep
        opt = opt[:-len(sep)]
        opt += "\n"
    print(opt)
    return opt


if __name__ == "__main__":
    _rows = [
        dict(epochs=1),
    ]
    _cols = [
        dict(factors=20),
        dict(factors=40),
    ]
    _data_name = "solar"
    _base_args = dict()
    print_res(_rows, _cols, _data_name, _base_args)

