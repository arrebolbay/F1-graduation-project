# -*- coding: utf-8 -*-
r"""
把 F1毕设 里的 CSV 拆解成一行行数据，导入 MySQL。

一、数据从哪里来？
    F1毕设\<年份>\<车手缩写>\
        <车手小写>_monaco_<年份>_all_laps_telemetry.csv   遥测：每隔约 0.1 秒一条
        <车手小写>_monaco_<年份>_lap_status.csv            圈状态：每圈一条
        lap_summary.csv                                    圈汇总：每圈一条

二、导入到哪里？
    数据库: F1-graduation-project（名字里有减号，所以 SQL 里必须用反引号 ` 包起来）
    数据表:
        telemetry    逐采样点遥测（数据量最大，约 360 万行）
        lap_status   逐圈状态（轮胎、进站、圈速等）
        lap_summary  逐圈汇总（速度、距离、异常圈标记等）
        v_lap_overview  视图：把 lap_status 和 lap_summary 按“年份+车手+圈号”拼在一起

三、怎么运行？（不需要先 activate，直接指定 test 环境的 python）
    D:\Anaconda3\envs\test\python.exe "G:\C\F1毕设\脚本\import_mysql.py"

    只想导入某几年：
    D:\Anaconda3\envs\test\python.exe "G:\C\F1毕设\脚本\import_mysql.py" --years 2024 2025

    想删掉旧表、从头再来：
    D:\Anaconda3\envs\test\python.exe "G:\C\F1毕设\脚本\import_mysql.py" --rebuild

四、重复运行会怎样？
    脚本会先删掉“本次要导入的年份+车手”的旧数据，再插入新数据。
    所以同一个命令跑第二遍不会产生重复行；其他年份的数据也不会被动到。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import pymysql


# ============================================================================
# 一、数据库连接配置
# ============================================================================
# 取值顺序(与 web/engine.py 的 AI 密钥同一套约定):
#   1) 环境变量 MYSQL_HOST / MYSQL_PORT / MYSQL_USER / MYSQL_PASSWORD / MYSQL_DATABASE
#   2) 本地私有文件 脚本/db_config.local.json(已写入 .gitignore,严禁入库)
#   3) 非敏感项(host/port/user/database)有默认值;
#      **密码没有默认值** —— 缺失时直接给出可操作的报错,绝不把真实密码写进代码
#      (旧版本在第 56 行用 os.getenv 的兜底值写着明文密码,一旦提交推送即泄漏)。
SECRET_FILE = Path(__file__).resolve().parent / "db_config.local.json"


def _load_local_secrets():
    """读取 脚本/db_config.local.json;文件不存在或格式错误时返回空字典。"""
    if not SECRET_FILE.exists():
        return {}
    try:
        return json.loads(SECRET_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


_LOCAL_SECRETS = _load_local_secrets()


def _cfg(key, env_name, default=None):
    """非敏感配置: 环境变量 → 本地文件 → 默认值。"""
    for v in (os.getenv(env_name), _LOCAL_SECRETS.get(key)):
        if v is not None and str(v).strip():
            return v
    return default


def _password():
    """密码: 环境变量 → 本地文件,两者都没有就明确报错(不静默使用默认值)。"""
    for v in (os.getenv("MYSQL_PASSWORD"), _LOCAL_SECRETS.get("password")):
        if v is not None and str(v).strip():
            return str(v).strip()
    sys.exit(
        "缺少数据库密码,二选一:\n"
        "  1) 设置环境变量:  set MYSQL_PASSWORD=你的密码\n"
        f"  2) 新建本地配置文件: {SECRET_FILE}\n"
        f"     内容格式参考同目录的 {SECRET_FILE.name}.example(该文件不入库)"
    )


DB_CONFIG = {
    "host": _cfg("host", "MYSQL_HOST", "localhost"),
    "port": int(_cfg("port", "MYSQL_PORT", 3307)),   # 安装 MySQL 时改过端口,所以是 3307
    "user": _cfg("user", "MYSQL_USER", "arrebol"),
    "password": _password(),
    "charset": "utf8mb4",                            # 支持中文和 emoji
    "autocommit": False,                             # 手动 commit，出错可以 rollback
}

DEFAULT_DATABASE = _cfg("database", "MYSQL_DATABASE", "F1-graduation-project")

# 每次 executemany 提交多少行。5000 是速度和内存之间的平衡点。
DEFAULT_BATCH_SIZE = 5000

# 脚本所在目录: ...\F1毕设\脚本
SCRIPT_DIR = Path(__file__).resolve().parent
# 项目根目录:   ...\F1毕设
PROJECT_DIR = SCRIPT_DIR.parent


# ============================================================================
# 二、建库 / 建表 SQL
# ============================================================================
# 说明：
#   - 所有字段名用英文，方便写 SQL；中文字段含义写在 COMMENT 里。
#   - year + driver 用来区分“哪一年、哪位车手”，三张表都有。
#   - id 是自增主键，仅用于唯一标识一行，业务上不需要关心。
CREATE_TABLE_SQL = {
    "telemetry": """
        CREATE TABLE IF NOT EXISTS `telemetry` (
            `id` BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT '自增主键',
            `year` SMALLINT UNSIGNED NOT NULL COMMENT '赛季年份',
            `driver` CHAR(3) NOT NULL COMMENT '车手三字母缩写，如 LEC',
            `session_date` DATETIME(6) NULL COMMENT '遥测采样时刻',
            `session_time_s` DOUBLE NULL COMMENT '距 session 开始多少秒',
            `driver_ahead` INT NULL COMMENT '前方车手的编号',
            `distance_to_driver_ahead` DOUBLE NULL COMMENT '与前车的距离',
            `time_s` DOUBLE NULL COMMENT '本圈已经行驶了多少秒',
            `rpm` DOUBLE NULL COMMENT '发动机转速',
            `speed` DOUBLE NULL COMMENT '速度 km/h',
            `n_gear` TINYINT UNSIGNED NULL COMMENT '档位',
            `throttle` DOUBLE NULL COMMENT '油门百分比 0-100',
            `brake` TINYINT(1) NULL COMMENT '是否踩刹车：1 是，0 否',
            `drs` TINYINT UNSIGNED NULL COMMENT 'DRS 状态码',
            `source` VARCHAR(32) NULL COMMENT '数据来源 car/pos/interpolation',
            `distance` DOUBLE NULL COMMENT '本圈累计行驶距离 m',
            `relative_distance` DOUBLE NULL COMMENT '本圈相对位置 0-1',
            `status` VARCHAR(32) NULL COMMENT '遥测状态，如 OnTrack',
            `x` DOUBLE NULL COMMENT '赛道坐标 X',
            `y` DOUBLE NULL COMMENT '赛道坐标 Y',
            `z` DOUBLE NULL COMMENT '赛道坐标 Z',
            `lap_number` SMALLINT UNSIGNED NULL COMMENT '第几圈',
            PRIMARY KEY (`id`),
            KEY `idx_year_driver` (`year`, `driver`)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
          COMMENT='逐采样点遥测数据（all_laps_telemetry.csv）'
    """,
    "lap_status": """
        CREATE TABLE IF NOT EXISTS `lap_status` (
            `id` BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT '自增主键',
            `year` SMALLINT UNSIGNED NOT NULL COMMENT '赛季年份',
            `driver` CHAR(3) NOT NULL COMMENT '车手三字母缩写',
            `lap_number` SMALLINT UNSIGNED NULL COMMENT '第几圈',
            `track_status` VARCHAR(16) NULL COMMENT '赛道状态码，1 表示全绿',
            `stint` SMALLINT UNSIGNED NULL COMMENT '第几段轮胎 stint',
            `tyre_life` DOUBLE NULL COMMENT '轮胎已使用圈数',
            `fresh_tyre` TINYINT(1) NULL COMMENT '是否新胎：1 是，0 否',
            `compound` VARCHAR(16) NULL COMMENT '轮胎配方 SOFT/MEDIUM/HARD/INTERMEDIATE/WET',
            `pit_in_time_s` DOUBLE NULL COMMENT '进站时刻（距 session 开始秒数）',
            `pit_out_time_s` DOUBLE NULL COMMENT '出站时刻（距 session 开始秒数）',
            `lap_time_s` DOUBLE NULL COMMENT '圈速（秒）',
            PRIMARY KEY (`id`),
            KEY `idx_year_driver` (`year`, `driver`)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
          COMMENT='逐圈状态（*_lap_status.csv）'
    """,
    "lap_summary": """
        CREATE TABLE IF NOT EXISTS `lap_summary` (
            `id` BIGINT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT '自增主键',
            `year` SMALLINT UNSIGNED NOT NULL COMMENT '赛季年份',
            `driver` CHAR(3) NOT NULL COMMENT '车手三字母缩写',
            `lap_number` SMALLINT UNSIGNED NULL COMMENT '圈号',
            `valid_duration_s` DOUBLE NULL COMMENT '有效行驶时长（秒）',
            `max_speed_kmh` DOUBLE NULL COMMENT '本圈最高时速 km/h',
            `avg_speed_kmh` DOUBLE NULL COMMENT '本圈平均时速 km/h',
            `distance_m` DOUBLE NULL COMMENT '本圈行驶距离 m',
            `pit_stop_excluded_s` DOUBLE NULL COMMENT '被剔除的停车时间（秒）',
            `lap_status` VARCHAR(16) NULL COMMENT '原始圈状态编码（部分年份没有此列）',
            `is_anomaly_lap` TINYINT(1) NULL COMMENT '是否异常圈：1 是，0 否',
            `anomaly_reason` VARCHAR(64) NULL COMMENT '异常原因（部分年份没有此列）',
            `fit_included` TINYINT(1) NULL COMMENT '是否参与拟合：1 是，0 否',
            `tyre_age_lap` SMALLINT NULL COMMENT '胎龄（圈）',
            `telemetry_rows` INT NULL COMMENT '该圈遥测原始行数',
            PRIMARY KEY (`id`),
            KEY `idx_year_driver` (`year`, `driver`)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
          COMMENT='逐圈汇总（lap_summary.csv）'
    """,
}

# 这个视图把两张“每圈一条”的表拼起来，小白查数据时不用自己写 JOIN。
CREATE_VIEW_SQL = """
    CREATE OR REPLACE VIEW `v_lap_overview` AS
    SELECT
        s.`year`,
        s.`driver`,
        s.`lap_number`,
        s.`compound`,
        s.`track_status`,
        s.`tyre_life`,
        s.`lap_time_s`,
        m.`valid_duration_s`,
        m.`max_speed_kmh`,
        m.`avg_speed_kmh`,
        m.`is_anomaly_lap`,
        m.`anomaly_reason`,
        m.`telemetry_rows`
    FROM `lap_status` AS s
    LEFT JOIN `lap_summary` AS m
           ON m.`year` = s.`year`
          AND m.`driver` = s.`driver`
          AND m.`lap_number` = s.`lap_number`
"""


# ============================================================================
# 三、CSV 列 -> 数据库列 的对应关系
# ============================================================================
# 数据库里一律用英文列名；原始 CSV 里的中文列名在 prepare_lap_summary() 里映射。
TELEMETRY_COLUMNS = [
    "year", "driver", "session_date", "session_time_s", "driver_ahead",
    "distance_to_driver_ahead", "time_s", "rpm", "speed", "n_gear",
    "throttle", "brake", "drs", "source", "distance", "relative_distance",
    "status", "x", "y", "z", "lap_number",
]

LAP_STATUS_COLUMNS = [
    "year", "driver", "lap_number", "track_status", "stint", "tyre_life",
    "fresh_tyre", "compound", "pit_in_time_s", "pit_out_time_s", "lap_time_s",
]

LAP_SUMMARY_COLUMNS = [
    "year", "driver", "lap_number", "valid_duration_s", "max_speed_kmh",
    "avg_speed_kmh", "distance_m", "pit_stop_excluded_s", "lap_status",
    "is_anomaly_lap", "anomaly_reason", "fit_included", "tyre_age_lap",
    "telemetry_rows",
]


# ============================================================================
# 四、小工具函数
# ============================================================================
def log(message: str = "") -> None:
    """打印一行并立刻刷新，这样在终端里能看到实时进度。"""
    print(message, flush=True)


def connect(database: str | None = None) -> pymysql.connections.Connection:
    """连接 MySQL。传 database=None 时只连服务器，用来执行 CREATE DATABASE。"""
    kwargs = dict(DB_CONFIG)
    if database:
        kwargs["database"] = database
    return pymysql.connect(**kwargs)


def to_float(series: pd.Series) -> pd.Series:
    """把一列文本转成小数；转不了的值变成 NaN（写库时再变成 NULL）。"""
    return pd.to_numeric(series, errors="coerce").astype("float64")


def to_int(series: pd.Series) -> pd.Series:
    """
    把一列数字转成“可空整数”。
    pandas 的 Int64（首字母大写）支持缺失值，普通 int 不支持。
    """
    return pd.to_numeric(series, errors="coerce").round().astype("Int64")


def to_text(series: pd.Series) -> pd.Series:
    """转成文本并去掉首尾空格。"""
    return series.astype("string").str.strip()


def to_seconds(series: pd.Series) -> pd.Series:
    """
    把 FastF1 的时间格式 '0 days 00:00:00.155000' 转成秒数 0.155。
    空值会变成 NaN。
    """
    return pd.to_timedelta(series.astype("string"), errors="coerce").dt.total_seconds()


def to_code_text(series: pd.Series) -> pd.Series:
    """
    状态码统一成文本。
    例如 pandas 把 TrackStatus 读成整数 12，这里仍然存成字符串 '12'；
    如果本来就是 '1|2' 这种文本，就原样保留。
    """
    text = series.astype("string").str.strip()
    number = pd.to_numeric(text, errors="coerce")
    normalized = number.round().astype("Int64").astype("string")
    return text.where(number.isna(), normalized)


def yes_no_to_int(series: pd.Series) -> pd.Series:
    """把中文的 是/否 转成 1/0，其他值变成空。"""
    return to_text(series).map({"是": 1, "否": 0}).astype("Int64")


def optional_column(raw: pd.DataFrame, name: str) -> pd.Series:
    """
    取一列；如果这份 CSV 没有这一列（老格式没有“异常原因”等字段），
    就返回整列空值，保证后续代码不用写一堆 if。
    """
    if name in raw.columns:
        return raw[name]
    return pd.Series(pd.NA, index=raw.index, dtype="object")


# ============================================================================
# 五、把三种 CSV 整理成“数据库表的样子”
# ============================================================================
def prepare_telemetry(raw: pd.DataFrame, year: int, driver: str) -> pd.DataFrame:
    """遥测 CSV：一行 = 一个采样点。"""
    out = pd.DataFrame(index=raw.index)

    # ---- 身份信息：从目录名得到年份和车手，比 CSV 里的列更可靠 ----
    out["year"] = year
    out["driver"] = driver

    # ---- 时间列 ----
    # Date 是真实时钟时间；存成 'YYYY-MM-DD HH:MM:SS.ffffff' 文本，MySQL 能直接识别。
    stamp = pd.to_datetime(raw["Date"], errors="coerce")
    out["session_date"] = stamp.dt.strftime("%Y-%m-%d %H:%M:%S.%f")
    out["session_time_s"] = to_seconds(raw["SessionTime"])
    out["time_s"] = to_seconds(raw["Time"])

    # ---- 数值列 ----
    out["driver_ahead"] = to_int(raw["DriverAhead"])
    out["distance_to_driver_ahead"] = to_float(raw["DistanceToDriverAhead"])
    out["rpm"] = to_float(raw["RPM"])
    out["speed"] = to_float(raw["Speed"])
    out["n_gear"] = to_int(raw["nGear"])
    out["throttle"] = to_float(raw["Throttle"])

    # Brake 在 CSV 里是 True/False，数据库里存 1/0 更省空间。
    out["brake"] = raw["Brake"].map({True: 1, False: 0}).astype("Int64")
    out["drs"] = to_int(raw["DRS"])

    # ---- 文本列 ----
    out["source"] = to_text(raw["Source"])
    out["status"] = to_text(raw["Status"])

    # ---- 坐标 / 距离 ----
    out["distance"] = to_float(raw["Distance"])
    out["relative_distance"] = to_float(raw["RelativeDistance"])
    out["x"] = to_float(raw["X"])
    out["y"] = to_float(raw["Y"])
    out["z"] = to_float(raw["Z"])
    out["lap_number"] = to_int(raw["LapNumber"])

    return out[TELEMETRY_COLUMNS]


def prepare_lap_status(raw: pd.DataFrame, year: int, driver: str) -> pd.DataFrame:
    """圈状态 CSV：一行 = 一圈。"""
    out = pd.DataFrame(index=raw.index)

    out["year"] = year
    out["driver"] = driver
    out["lap_number"] = to_int(raw["LapNumber"])
    out["track_status"] = to_code_text(raw["TrackStatus"])
    out["stint"] = to_int(raw["Stint"])
    out["tyre_life"] = to_float(raw["TyreLife"])
    out["fresh_tyre"] = raw["FreshTyre"].map({True: 1, False: 0}).astype("Int64")
    out["compound"] = to_text(raw["Compound"])

    # PitInTime / PitOutTime 也是 '0 days ...' 格式，统一转成秒。
    out["pit_in_time_s"] = to_seconds(raw["PitInTime"])
    out["pit_out_time_s"] = to_seconds(raw["PitOutTime"])
    out["lap_time_s"] = to_float(raw["LapTime_s"])

    return out[LAP_STATUS_COLUMNS]


def prepare_lap_summary(raw: pd.DataFrame, year: int, driver: str) -> pd.DataFrame:
    """圈汇总 CSV：一行 = 一圈，列名是中文，这里逐个翻译成数据库列名。"""
    out = pd.DataFrame(index=raw.index)

    out["year"] = year
    out["driver"] = driver
    out["lap_number"] = to_int(raw["圈号"])
    out["valid_duration_s"] = to_float(raw["有效行驶时长_s"])
    out["max_speed_kmh"] = to_float(raw["最高时速_kmh"])
    out["avg_speed_kmh"] = to_float(raw["平均时速_kmh"])
    out["distance_m"] = to_float(raw["圈内行驶距离_m"])
    out["pit_stop_excluded_s"] = to_float(raw["剔除停车_s"])

    # 下面几列只有部分年份有；optional_column() 会在缺失时补一整列空值。
    out["lap_status"] = to_code_text(optional_column(raw, "圈状态"))
    out["is_anomaly_lap"] = yes_no_to_int(optional_column(raw, "是否异常圈"))
    out["anomaly_reason"] = to_text(optional_column(raw, "异常原因"))
    out["fit_included"] = yes_no_to_int(optional_column(raw, "参与拟合"))
    out["tyre_age_lap"] = to_int(optional_column(raw, "胎龄_圈"))
    out["telemetry_rows"] = to_int(optional_column(raw, "遥测行数"))

    return out[LAP_SUMMARY_COLUMNS]


# ============================================================================
# 六、写库
# ============================================================================
def insert_dataframe(cursor, table: str, frame: pd.DataFrame,
                     batch_size: int, label: str = "") -> int:
    """
    把 DataFrame 里的行分批写进 MySQL。

    为什么不用 df.to_sql()？
      to_sql 在几十万行时很慢；executemany 一次发送 5000 行，速度快很多。
    """
    if frame.empty:
        return 0

    columns = list(frame.columns)
    column_sql = ", ".join(f"`{name}`" for name in columns)
    placeholder_sql = ", ".join(["%s"] * len(columns))
    sql = f"INSERT INTO `{table}` ({column_sql}) VALUES ({placeholder_sql})"

    # astype(object) -> 让 pandas 放过类型检查；where(notna, None) -> 把 NaN/NaT 变成 NULL。
    cleaned = frame.astype(object).where(pd.notna(frame), None)
    rows = cleaned.to_numpy().tolist()

    total = len(rows)
    for start in range(0, total, batch_size):
        batch = rows[start:start + batch_size]
        cursor.executemany(sql, batch)
        done = min(start + batch_size, total)
        if total > batch_size:
            print(f"      {label} 已写入 {done:,}/{total:,} 行", end="\r", flush=True)
    if total > batch_size:
        print(" " * 60, end="\r")  # 擦掉进度行
    return total


def delete_driver_rows(cursor, year: int, driver: str) -> None:
    """删除某一年、某位车手在三张表里的旧数据，保证重复运行不会翻倍。"""
    for table in ("telemetry", "lap_status", "lap_summary"):
        cursor.execute(
            f"DELETE FROM `{table}` WHERE `year` = %s AND `driver` = %s",
            (year, driver),
        )


def ensure_index(cursor, database: str, table: str, index_name: str,
                 columns: list[str]) -> None:
    """
    如果索引不存在就创建。
    先导数据、后建索引，比一边插一边维护索引快很多。
    """
    cursor.execute(
        """
        SELECT 1
        FROM information_schema.statistics
        WHERE table_schema = %s AND table_name = %s AND index_name = %s
        LIMIT 1
        """,
        (database, table, index_name),
    )
    if cursor.fetchone() is None:
        column_sql = ", ".join(f"`{name}`" for name in columns)
        cursor.execute(f"ALTER TABLE `{table}` ADD INDEX `{index_name}` ({column_sql})")
        log(f"   已创建索引 {table}.{index_name}")


# ============================================================================
# 七、扫描目录
# ============================================================================
def discover_years(base_dir: Path) -> list[int]:
    """找出 F1毕设 下所有纯数字文件夹，如 2022、2023、2024、2025。"""
    years = []
    for child in base_dir.iterdir():
        if child.is_dir() and child.name.isdigit():
            years.append(int(child.name))
    return sorted(years)


def find_driver_tasks(base_dir: Path, years: list[int]):
    """
    逐个车手目录返回 (年份, 车手, 三个文件路径)。
    没有的文件返回 None，后面的循环会跳过。
    """
    for year in years:
        year_dir = base_dir / str(year)
        if not year_dir.is_dir():
            log(f"⚠️  找不到年份目录，已跳过: {year_dir}")
            continue

        for driver_dir in sorted(path for path in year_dir.iterdir() if path.is_dir()):
            driver = driver_dir.name.strip().upper()

            telemetry_file = None
            status_file = None
            summary_file = None

            for csv_file in driver_dir.iterdir():
                if not csv_file.is_file() or csv_file.suffix.lower() != ".csv":
                    continue
                name = csv_file.name.lower()
                if name.endswith("_all_laps_telemetry.csv"):
                    telemetry_file = csv_file
                elif name.endswith("_lap_status.csv"):
                    status_file = csv_file
                elif name == "lap_summary.csv":
                    summary_file = csv_file

            if telemetry_file is None and status_file is None and summary_file is None:
                continue

            yield year, driver, {
                "telemetry": telemetry_file,
                "lap_status": status_file,
                "lap_summary": summary_file,
            }


# ============================================================================
# 八、主流程
# ============================================================================
def main() -> int:
    parser = argparse.ArgumentParser(
        description="把 F1毕设 的 CSV 拆成数据行，导入 MySQL 的 F1-graduation-project 库。"
    )
    parser.add_argument(
        "--base-dir", default=str(PROJECT_DIR),
        help=f"F1毕设 目录，默认 {PROJECT_DIR}",
    )
    parser.add_argument(
        "--database", default=DEFAULT_DATABASE,
        help=f"数据库名，默认 {DEFAULT_DATABASE}",
    )
    parser.add_argument(
        "--years", nargs="+", type=int, default=None,
        help="只导入指定年份，可写多个，例如 --years 2024 2025；默认全部年份",
    )
    parser.add_argument(
        "--rebuild", action="store_true",
        help="先删除旧的三张表和视图，再重新创建（数据会全部丢失！）",
    )
    parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
        help=f"每批写入多少行，默认 {DEFAULT_BATCH_SIZE}",
    )
    args = parser.parse_args()

    base_dir = Path(args.base_dir).resolve()
    if not base_dir.is_dir():
        log(f"❌ 找不到数据目录: {base_dir}")
        return 2

    database = args.database
    years = args.years if args.years else discover_years(base_dir)
    if not years:
        log(f"❌ 在 {base_dir} 下没有找到 2022/2023/... 这样的年份文件夹")
        return 2

    log("=" * 72)
    log("F1 毕设 CSV -> MySQL 导入工具")
    log(f"数据目录  : {base_dir}")
    log(f"目标数据库: {database}")
    log(f"导入年份  : {', '.join(str(y) for y in years)}")
    log("=" * 72)

    # ---- 第 1 步：连接服务器并创建数据库 ----
    try:
        server_conn = connect()
        with server_conn.cursor() as cursor:
            # 数据库名带减号，必须用反引号包住，否则 MySQL 会把它当成减法。
            cursor.execute(
                f"CREATE DATABASE IF NOT EXISTS `{database}` "
                "DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
            )
        server_conn.commit()
        server_conn.close()
        log(f"✅ 数据库已就绪: {database}")
    except pymysql.MySQLError as exc:
        log(f"❌ 连接 MySQL 或建库失败: {exc}")
        log("   请确认 MySQL 服务已启动、端口/用户名/密码正确。")
        return 1

    # ---- 第 2 步：连接目标数据库，按需删表、建表 ----
    conn = connect(database)
    try:
        with conn.cursor() as cursor:
            if args.rebuild:
                log("🧨 --rebuild 已开启：删除旧视图和旧表 ...")
                cursor.execute("DROP VIEW IF EXISTS `v_lap_overview`")
                for table in ("telemetry", "lap_status", "lap_summary"):
                    cursor.execute(f"DROP TABLE IF EXISTS `{table}`")
                conn.commit()

            for table, ddl in CREATE_TABLE_SQL.items():
                cursor.execute(ddl)
            conn.commit()
            log("✅ 数据表已就绪: telemetry / lap_status / lap_summary")
            log("")

            # ---- 第 3 步：逐个车手导入 ----
            started_at = time.time()
            total_rows = {"telemetry": 0, "lap_status": 0, "lap_summary": 0}
            file_count = {"telemetry": 0, "lap_status": 0, "lap_summary": 0}
            failed = []
            task_count = 0

            for year, driver, files in find_driver_tasks(base_dir, years):
                task_count += 1
                log(f"▶ {year} / {driver}")

                # 读文件 + 写库放在同一个事务里：中途报错就 rollback，不会留下半份数据。
                try:
                    delete_driver_rows(cursor, year, driver)

                    for kind, path in files.items():
                        if path is None:
                            log(f"   - {kind:<11} 没有这个文件，跳过")
                            continue

                        # 统一用 utf-8-sig 读取：能自动吃掉 Windows Excel 可能带的 BOM。
                        raw = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
                        if raw.empty:
                            log(f"   - {kind:<11} 文件是空的，跳过: {path.name}")
                            continue

                        if kind == "telemetry":
                            prepared = prepare_telemetry(raw, year, driver)
                        elif kind == "lap_status":
                            prepared = prepare_lap_status(raw, year, driver)
                        else:
                            prepared = prepare_lap_summary(raw, year, driver)

                        written = insert_dataframe(
                            cursor, kind, prepared, args.batch_size,
                            label=kind,
                        )
                        total_rows[kind] += written
                        file_count[kind] += 1
                        log(f"   - {kind:<11} {written:>8,} 行  <- {path.name}")

                    conn.commit()
                except Exception as exc:  # 一个车手失败不影响其他人，最后统一报告
                    conn.rollback()
                    failed.append(f"{year}/{driver}: {exc}")
                    log(f"   ❌ 导入失败，已回滚: {exc}")

            # ---- 第 4 步：补建查询索引 ----
            log("")
            log("🔧 正在检查/创建查询索引 ...")
            ensure_index(cursor, database, "telemetry", "idx_year_driver_lap",
                         ["year", "driver", "lap_number"])
            ensure_index(cursor, database, "lap_status", "idx_year_driver_lap",
                         ["year", "driver", "lap_number"])
            ensure_index(cursor, database, "lap_summary", "idx_year_driver_lap",
                         ["year", "driver", "lap_number"])

            # ---- 第 5 步：创建方便查询的视图 ----
            cursor.execute(CREATE_VIEW_SQL)
            conn.commit()
            log("✅ 视图已就绪: v_lap_overview")

            # ---- 第 6 步：用数据库自己的计数做最终校验 ----
            log("")
            log("=" * 72)
            log("导入结果（以数据库实际行数为准）")
            log("=" * 72)
            for table in ("telemetry", "lap_status", "lap_summary"):
                cursor.execute(f"SELECT COUNT(*) FROM `{table}`")
                db_rows = cursor.fetchone()[0]
                log(f"  {table:<12} {db_rows:>10,} 行"
                    f"   （本次处理 {total_rows[table]:,} 行 / {file_count[table]} 个文件）")

            log("")
            log("按年份统计 telemetry：")
            cursor.execute(
                "SELECT `year`, COUNT(*) FROM `telemetry` "
                "GROUP BY `year` ORDER BY `year`"
            )
            for row_year, row_count in cursor.fetchall():
                log(f"  {row_year}: {row_count:,} 行")

            elapsed = time.time() - started_at
            log("")
            log(f"⏱  总耗时: {elapsed:.1f} 秒，处理车手目录 {task_count} 个")

            if failed:
                log("")
                log(f"⚠️  有 {len(failed)} 个车手导入失败：")
                for item in failed:
                    log(f"   - {item}")
                return 1

            log("🎉 全部导入完成！")
            return 0

    except pymysql.MySQLError as exc:
        conn.rollback()
        log(f"❌ 数据库操作失败: {exc}")
        return 1
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())


