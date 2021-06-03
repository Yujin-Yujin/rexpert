class Prefix():
    def __init__():
        self.prefix_dict={
            "isAfter":"happens after",
            "xReason":"because",
            "xNeed":"but before, PersonX needed",
            "xIntent":"because PersonX wanted",

            "causes": "causes",
            "xEffect": "as a result, PersonX will",
            "isBefore": "happens before",
            "xReact":"as a result, PersonX feels",
            "xWant":"as a result, PersonX wants",
            "oEffect":"as a result, Y or others will",
            "oReact":"as a result, Y or others feels",
            "oWant":"as a result, Y or others want"
        }
    def get_prefix(relation):
        return self.prefix_dict[relation]