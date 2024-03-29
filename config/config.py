DICT_CATEGORICAL = {
    'hightlighttag': ['坐班', '不装卸', '营业额奖', '提供电动车', '工资上不封顶', '无销售性质', '高薪', '可预支工资', '包吃包住', '五险一金'],
    'employer_tag': ['帅哥美女多', '做五休二', '八小时制', '做一休一', '过节福利', '就近安排', '坐班', '工资随走随结', '寒冷补贴', '免费维修',
                     '免费培训', '周末双休', '全勤奖', '员工宿舍', '五百强企业', '交社保', '老骑手带领', '法定节假日三薪', '带薪休假', '话补',
                     '绩效奖金', '无需经验', '冲单补贴', '做三休一', '薪资有保证', '加班费', '销售额提成', '晚班补贴', '保底工资', '两班倒',
                     '计件', '环境好', '年底双薪', '灵活班次', '奖金丰厚', '当天入住', '带薪培训', '五险二金', '老板好', '房补', '长夜班',
                     '计件提成', '做六休一', '五险', '宿舍好', '单量充足', '提供客户', '一人一车', '生日福利', '月休三天', '报销健康证',
                     '三班倒', '长白班', '按单提成', '晋升空间大', '计件工资', '不加班', '交金', '巡逻岗', '大额单补贴', '高温补贴',
                     '朝九晚五', '免费工装', '食宿补贴', '门店氛围好', '日结工资', '保镖', '不压工资', '只包住', '门卫', '做二休一',
                     '安检员', '计重提成', '报销体检费', '月休四天', '免费保养', '员工旅游', '可兼职', '底薪加提成', '报销春节往返路费',
                     '平台派单', '报销车票', '可带手机', '员工聚餐', '直招', '提供车', '餐补', '可开回家', '提成高', '形象岗', '弹性工作',
                     '时间自由', '多劳多得', '年终奖', '工龄奖', '当天提车', '准时发薪', '自带车补贴', '只包吃', '不体检', '交通补贴'],
    'distance': [i for i in range(13)],
    'salary_min': [i for i in range(13)],
    'active_tag': ['在线', '最近活跃', '刚刚活跃'],
    'gender': [0, 1, 2],
    'age': [i for i in range(13)],
    'new_channel_no': ['blbl', 'csj', 'douyin', 'else-', 'else-1', 'else-100', 'else-2', 'else-3', 'else-360', 'else-360s', 'else-5',
                       'else-6', 'else-7', 'else-8', 'else-9', 'else-HWXXL', 'else-abc', 'else-anzhi', 'else-baidu', 'else-bdxcx1',
                       'else-bdxcx2', 'else-bdxcx3', 'else-channel001', 'else-channel002', 'else-channel003', 'else-channel004',
                       'else-channel005', 'else-channel006', 'else-channel008', 'else-channel009', 'else-channel010', 'else-channel014',
                       'else-channel016', 'else-channel017', 'else-channel050', 'else-default', 'else-douyin', 'else-gdt1', 'else-gdt2',
                       'else-gdt3', 'else-gdt4', 'else-gdt6', 'else-gdt7', 'else-hj56', 'else-jrtt', 'else-kuaishou', 'else-kuaishou-k',
                       'else-kuaishou-z', 'else-kuaishou01', 'else-kuaishou2', 'else-lenove', 'else-meizu', 'else-mumayi', 'else-sogous',
                       'else-sougou', 'else-tencent', 'else-wzpush', 'else-xxff', 'else-yybs', 'else-yybs2', 'guangdiantong', 'h5',
                       'huawei', 'huaweis', 'huaweix', 'ios', 'jbp', 'jrtt', 'kbgz', 'kuaishou', 'oppo', 'vivo', 'weixinmp', 'xiaomi',
                       'xiaoyu', 'yyb', 'yybs'],
    'fast_job_status': [1, 2],
    'city_id': [11000, 12000, 13001, 13002, 13003, 13004, 13005, 13006, 13007, 13008, 13009, 13010, 13011, 14001, 14002, 14003, 14004,
                14005, 14006, 14007, 14008, 14009, 14010, 14011, 15001, 15002, 15003, 15004, 15005, 15006, 15007, 15008, 15009, 15022,
                15025, 15029, 21001, 21002, 21003, 21004, 21005, 21006, 21007, 21008, 21009, 21010, 21011, 21012, 21013, 21014, 22001,
                22002, 22003, 22004, 22005, 22006, 22007, 22008, 22024, 23001, 23002, 23003, 23004, 23005, 23006, 23007, 23008, 23009,
                23010, 23011, 23012, 23027, 31000, 32001, 32002, 32003, 32004, 32005, 32006, 32007, 32008, 32009, 32010, 32011, 32012,
                32013, 33001, 33002, 33003, 33004, 33005, 33006, 33007, 33008, 33009, 33010, 33011, 34001, 34002, 34003, 34004, 34005,
                34006, 34007, 34008, 34010, 34011, 34012, 34013, 34015, 34016, 34017, 34018, 35001, 35002, 35003, 35004, 35005, 35006,
                35007, 35008, 35009, 36001, 36002, 36003, 36004, 36005, 36006, 36007, 36008, 36009, 36010, 36011, 37001, 37002, 37003,
                37004, 37005, 37006, 37007, 37008, 37009, 37010, 37011, 37012, 37013, 37014, 37015, 37016, 37017, 41001, 41002, 41003,
                41004, 41005, 41006, 41007, 41008, 41009, 41010, 41011, 41012, 41013, 41014, 41015, 41016, 41017, 41018, 42001, 42002,
                42003, 42005, 42006, 42007, 42008, 42009, 42010, 42011, 42012, 42013, 42028, 42029, 42094, 42095, 42096, 43001, 43002,
                43003, 43004, 43005, 43006, 43007, 43008, 43009, 43010, 43011, 43012, 43013, 43031, 44001, 44002, 44003, 44004, 44005,
                44006, 44007, 44008, 44009, 44012, 44013, 44014, 44015, 44016, 44017, 44018, 44019, 44020, 44051, 44052, 44053, 45001,
                45002, 45003, 45004, 45005, 45006, 45007, 45008, 45009, 45010, 45011, 45012, 45013, 45014, 46001, 46002, 46003, 46025,
                46026, 46027, 46028, 46030, 46031, 46033, 46034, 46035, 46036, 46091, 46092, 46093, 46095, 46096, 46097, 50000, 51001,
                51003, 51004, 51005, 51006, 51007, 51008, 51009, 51010, 51011, 51013, 51014, 51015, 51016, 51017, 51018, 51019, 51020,
                51032, 51033, 51034, 52001, 52002, 52003, 52004, 52022, 52023, 52024, 52026, 52027, 53001, 53003, 53004, 53005, 53006,
                53007, 53008, 53009, 53023, 53025, 53026, 53028, 53029, 53031, 53033, 53034, 54001, 54021, 54022, 54023, 54024, 54025,
                54026, 61001, 61002, 61003, 61004, 61005, 61006, 61007, 61008, 61009, 61010, 62001, 62002, 62003, 62004, 62005, 62006,
                62007, 62008, 62009, 62010, 62011, 62012, 62029, 62030, 63001, 63021, 63022, 63023, 63025, 63026, 63027, 63028, 64001,
                64002, 64003, 64004, 64005, 65001, 65002, 65021, 65022, 65023, 65027, 65028, 65029, 65030, 65031, 65032, 65040, 65042,
                65043, 65091, 65092, 65093, 65094, 65095, 65096, 65097, 65098, 65099, 65100, 71001, 71002, 71003, 71004, 71005, 71006,
                71007, 71008, 71009, 71010, 71011, 71012, 71013, 71014, 71015, 71016, 71017, 71018, 71019, 71020, 71021, 71022, 71023,
                71024, 71025, 81000, 82000],
    'category_id': [i for i in range(1, 500)]
}
BOUNDARIES = {
    'distance': [1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 40],
    'salary': [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 15000, 20000],
    'age': [0, 16, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65]
}
COL_NAME = ['label', 'user_id', 'job_id', 'distance', 'employer_tag', 'hightlighttag', 'salary_min', 'salary_max', 'active_tag',
            'company_id', 'gender', 'age', 'marriage', 'new_channel_no', 'expect_salary_min', 'expect_salary_max', 'fast_job_status',
            'expect_job', 'accommodation_schedule', 'city_id', 'past_experience', 'category_id']
DEFAULT_VALUE = [[0], [0], [0], [0.0], ['0'], ['0'], [0], [0], ['0'], [0], [0], [0], [0], ['0'], [0], [0], [0], ['0'], [0], [0], ['0'], [0]]
MASKS = [0, 0, 0, 0.0, '0', '0', 0, 0, '0', 0, 0, 0, 0, '0', 0, 0, 0, 0, 0, 0, 0, 0]
MASK_VALUE = dict(zip(COL_NAME[1:], MASKS[1:]))
