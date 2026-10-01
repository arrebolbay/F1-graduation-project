# F1 摩纳哥 GP · 停站策略优化系统(2022–2025)

**F1 Monaco Grand Prix Pit-Stop Strategy Optimization System** —— 基于真实 FastF1 遥测数据的
赛前最优停站策略(动态规划)+ 赛中实时决策(Stackelberg 博弈 + 蒙特卡洛)+ 零依赖 Web 演示系统。

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![FastF1](https://img.shields.io/badge/data-FastF1%202022--2025-e10600.svg)](https://github.com/theOehrly/FastF1)

> **English brief** — Pre-race optimal pit strategy via dynamic programming; mid-race decision-making
> via a Stackelberg game with Monte-Carlo simulation; per-driver tire degradation and lap-time
> volatility models fitted on real FastF1 telemetry of the 2022–2025 Monaco Grands Prix; plus a
> zero-dependency web app for interactive what-if analysis. Released under the [MIT License](LICENSE).

**毕业设计背景** —— 问题一(赛前): 在"至少使用两种干地配方"、最短 stint、正赛轮胎套数等规则约束下,
以动态规划为每位车手求解最优(或强制两停)停站策略;问题二(赛中): 以 Stackelberg 博弈刻画
"先进站 vs 跟进"的策略互动,蒙特卡洛(每圈圈速 ~ N(确定性圈速, σ²),σ 由实测标定)输出
**成功率、被反超风险与超越圈数分布**,并支持红旗(免费换胎 + 静态发车)/ SC·VSC 等旗种场景与 AI 策略分析。

## 快速开始

```powershell
# 启动 Web 演示系统(零第三方依赖,离线可用)
cd G:\C\F1毕设\web
D:\Anaconda3\envs\F1_graduation_project\python.exe server.py --open
# 浏览器访问 http://127.0.0.1:8765/ (端口被占自动顺延)
# 也可直接双击 web\启动前端.bat
```

模型与数据脚本见下文「四、使用方式」;Web 系统详见「七、Web 交互系统」。

`G:\C` 下所有与 F1 相关的内容已于 **2026-09-16** 集中整理到本文件夹 `G:\C\F1毕设`。
原位置(`G:\C` 顶层)已不再保留 F1 相关文件。

## 一、目录结构(按"年份 → 车手"两级存放)

```
F1毕设\
├─ README.md / 任务清单.md / LICENSE(MIT) / 推送更新.bat
├─ 脚本\                           全部 Python 脚本(通用工具,不按年份)
│   ├─ f1test.py                   FastF1 下载指定车手各圈遥测并导出 CSV
│   ├─ F1test2.py                  维斯塔潘 vs 勒克莱尔 正赛圈速对比图
│   ├─ csvtest.py                  CSV 数据概况速览(pandas)
│   ├─ plot_lec_monaco_speed.py    每圈时速曲线绘图(单圈/热力图/对比)
│   ├─ batch_export.py             批量下载四年 × 全车手遥测并出图
│   ├─ fit_tire_degradation.py     轮胎衰减拟合(汇总 C3/C4/C5 + 分车手)
│   ├─ fit_lap_noise.py            圈速波动 σ 估计 + 正态性检验
│   ├─ solve_dp_strategy.py        DP 赛前最优策略(1停/强制两停)
│   └─ solve_stackelberg.py        Stackelberg 蒙特卡洛(离线版)
├─ web\                            Web 交互系统(Python 标准库后端 + 原生前端)
│   ├─ server.py / engine.py       HTTP 服务 / 核心引擎(DP+MC+轮胎库存+AI)
│   ├─ 启动前端.bat                 一键启动
│   └─ static\                     index.html / style.css / app.js
├─ 图表\                           汇总图表与结果表(06~14 系列 CSV/PNG)
├─ 2022\ … 2025\                   年份文件夹(直接在 F1毕设\ 下)
│   └─ LEC\ … VER\                 车手三字母码
│       ├─ lec_monaco_2024_all_laps_telemetry.csv   遥测原始数据(~13 MB/车手)
│       ├─ 01_全部圈速度曲线叠加.png … 05_*.png      5 张汇总图
│       ├─ lap_summary.csv                           每圈统计表
│       └─ laps\lap_01.png … lap_78.png              单圈时速曲线图
└─ 缓存\f1_cache\                  FastF1 缓存(不入库,联网可重建)
```

> 数据规模: 76 份遥测 CSV(4 年 × 19 位车手)+ 5600+ 张图表;`缓存\` 与
> `web\history.jsonl`、`__pycache__` 等运行时产物不入库(见 `.gitignore`)。

## 二、命名规则

| 层级 | 约定 | 示例 |
| --- | --- | --- |
| 年份文件夹 | 4 位年份 | `2024`、`2025` |
| 车手文件夹 | FIA 三字母码(大写) | `LEC`、`VER`、`HAM`、`PIA` |
| 遥测 CSV 文件名 | `<车手小写>_monaco_<年份>_all_laps_telemetry.csv` | `lec_monaco_2024_all_laps_telemetry.csv` |
| 其它车手 | 改参数 `DRIVER` 后运行 `f1test.py` 自动输出到 `2024\VER\` 等 | — |

新增一位车手的一年数据,只需:
```powershell
cd F1毕设\脚本
# 修改 f1test.py 里的 YEAR 和 DRIVER,运行即可,CSV 自动落到 F1毕设\<年份>\<车手>\ 下
python f1test.py
```

## 三、环境依赖

| 用途 | 依赖 |
| --- | --- |
| 绘图 / 读 CSV | Python 3.12 + pandas + numpy + matplotlib(本机已具备) |
| 重新下载遥测(`f1test.py`、`F1test2.py`) | 额外需要 `pip install fastf1`(本机当前**未安装**) |

中文字体使用 Microsoft YaHei(脚本会自动探测,已确认无缺字警告)。

## 四、使用方式

```powershell
cd G:\C\F1毕设\脚本

# 绘制 2024 摩纳哥勒克莱尔时速曲线(图表自动输出到 2024\LEC\ 目录)
python plot_lec_monaco_speed.py                # 5 张汇总图 + lap_summary.csv
python plot_lec_monaco_speed.py --per-lap      # 额外 77 张单圈 PNG
python plot_lec_monaco_speed.py --x time       # 横轴换成"圈内有效行驶时间(s)"
python csvtest.py                              # 快速查看 CSV 结构

python f1test.py                               # (需 fastf1)抓取并输出到 2024\LEC\
python F1test2.py                              # (需 fastf1)生成 2024\VER与LEC圈速对比.png
```

所有脚本路径都基于**脚本自身位置**计算,图表输出到 CSV 所在目录,与运行时工作目录无关。
也可显式指定:`--csv 路径` / `--outdir 路径`。

## 五、数据说明与已知注意事项

1. **圈数**:CSV 含 77 圈(圈号 1、3~78),**没有第 2 圈**——2024 摩纳哥站第 1 圈发生
   严重事故,红旗中断后重新发车,数据源本身缺该圈。
2. **第 1 圈是红旗圈**:遥测片段长 2456 s(正常圈约 76~85 s),其中约 2250 s 赛车静止
   (停在 3201 m 处等待重新发车),共 18,711 行(正常圈约 600 行)。
   `plot_lec_monaco_speed.py` 会自动剔除"时速 < 2 km/h 且持续 > 15 s"的静止段,
   把该圈还原成接近正常一圈的形态,并在图与统计表中单独标注。
3. **坐标与单位**:`Speed` 单位 km/h(全场最大 288 km/h);`Distance` 已按圈重置,绘图前再
   归零,横轴即"圈内行驶距离 0~约 3300 m";`Time` 为 `"0 days 00:01:18.733000"` 形式的
   时间差字符串,脚本用 `pd.to_timedelta` 解析。
4. **实测结论**(2026-09-16 运行):最快圈第 71 圈 75.29 s;最慢正常圈第 3 圈 84.75 s
   (红旗后安全车带队);有效圈平均 78.57 s;全场最高时速 288 km/h。
5. 复现提示:`f1test.py` 需要联网下载 FastF1 数据,缓存命中时会直接使用 `缓存\f1_cache`。

## 六、整理时做过的路径调整

为避免"文件搬家后脚本跑不起来",整理时同步修改了:

- **整体结构**:选定"年份 → 车手"两级目录(`2024\LEC\…`)并去掉中间的 `数据\`、`图表\` 层,
  图表与 CSV 放在同一目录(方便查阅)。
- `plot_lec_monaco_speed.py`:路径解析改为搜索 `F1毕设\*\<车手>\` 格式,`--outdir` 留空时
  图表输出到 CSV 所在目录(自动镜像年份/车手);`DATA_DIR`/`CHART_DIR` 常量已移除,改为
  `PROJECT_DIR` + `glob` 通配。
- `f1test.py`:缓存目录由相对路径 `f1_cache` 改为 `缓存\f1_cache`,CSV 输出改为
  `F1毕设\<年份>\<车手>\`。
- `F1test2.py`:缓存目录同步改为 `缓存\f1_cache`,多年份对比图输出到 `F1毕设\<年份>\`,并补上
  中文字体设置。
- `csvtest.py`:CSV 路径改为自动搜索 `F1毕设\*\<车手>\lec_monaco_*_telemetry.csv`;支持命令行
  参数指定 CSV 路径。

> 注: 本文件夹 `F1毕设\` 自 2026-10-01 起为独立 Git 仓库,
> 远端 <https://github.com/arrebolbay/F1-graduation-project>。
> 后续新增年份/车手时,只需创建 `F1毕设\<年份>\<车手>\` 文件夹并运行对应脚本即可。

## 七、Web 交互系统(2026-09-29 新增,2026-10-01 扩展)

毕业设计演示前端,位于 `web\`,**零第三方依赖**(Python 标准库后端 + 原生 HTML/CSS/JS,离线可用):

```
F1毕设\web\
├─ server.py          Web 服务(标准库 http.server,含全部 API 路由)
├─ engine.py          核心引擎(数据加载 / 蒙特卡洛 / DP结果解析 / 历史记录)
├─ history.jsonl      历史记录(运行后自动生成)
├─ 启动前端.bat        一键启动并打开浏览器
└─ static\
   ├─ index.html      页面(赛前模拟 / 赛中决策 / 历史记录 三模块)
   ├─ style.css       F1 暗色主题(黑底红点缀,参考 F1 官网与实时计时仪表盘风格)
   └─ app.js          交互逻辑 + 手绘 SVG 图表(无 CDN,离线可用)
```

**启动方式(二选一):**

```powershell
# 方式一: 双击 web\启动前端.bat
# 方式二: 命令行
cd G:\C\F1毕设\web
D:\Anaconda3\envs\F1_graduation_project\python.exe server.py --open   # 自动打开浏览器
# 浏览器访问 http://127.0.0.1:8765/  (端口可用 --port 修改)
```

**四大模块:**

| 模块 | 输入 | 输出 |
|---|---|---|
| ① 赛前模拟 | 车手下拉 + 策略模式(最优 / 强制两停) + 轮胎库存(正赛至多 7 套) | 策略时间线、分段明细(含套装标签)、累计用时曲线、正常vs两停对比、练习/排位磨损代价 |
| ② 赛中决策 | 场景四选一(Undercut / Overcut / SC·VSC / 红旗) + 双方车手/胎型/胎龄 + 当前圈数/差距(秒) + 进站换上套装 | **每个响应分支的成功率 + 被反超风险(曾领先被反超率/位置翻转率/风险等级) + 若成功约几圈完成超越**、超越圈数分布直方图、期望时间差、轮胎配方选择、AI 策略分析 |
| ③ 数据总览 | — | 07b/07c/10b/10c/10d/11 等结果表站内渲染 |
| ④ 历史记录 | — | 全部请求记录(服务端持久化),支持一键**回填**表单重算、清空 |

**API 路由**(供二次开发):

| 路由 | 说明 |
|---|---|
| `GET /api/meta` | 车手/配方/场景/默认参数/轮胎模板/AI 配置状态 |
| `GET /api/prerace?driver=LEC&mode=normal\|two_stop&log=0` | 赛前 DP 策略(log=0 不写历史) |
| `POST /api/prerace` | 赛前 DP 策略(JSON: driver/mode/tires,轮胎库存约束求解) |
| `POST /api/simulate` | 赛中 Stackelberg 蒙特卡洛(JSON 请求体,支持轮胎库存、进站换上套装、旗种场景、参数覆盖) |
| `POST /api/ai` | AI 策略分析(传入 simulate 结果;配置 F1_AI_API_KEY 走大模型,否则本地规则分析) |
| `GET /api/tables` | 全部结果数据表 |
| `GET /api/history` / `POST /api/history/clear` | 历史记录读取 / 清空 |
| `GET /charts/*` | 直接引用 `图表\` 目录的 PNG |

**说明与已知边界:**

- 蒙特卡洛模型与 `脚本\solve_stackelberg.py` 同构(性能保持率衰减、进站混合分布、分场景分支语义),
  差异:按车手取基准圈速(07c)与分配方衰减参数(07b)、轮胎库存约束、参数可覆盖。
- **概率口径**: 每圈圈速 ~ N(确定性圈速, σ²);σ 由 `图表\14_圈速波动估计.csv` 实测标定
  (FastF1 2022-2025 摩纳哥正赛 stint 去趋势残差,1.5IQR 去尾后 σ≈1.05s,正态检验 p=0.09),
  并按车手波动比值加权。因此成功率/被反超风险是**连续概率**,不是硬判定。
- **轮胎规则口径**: 每车手 13 套干胎 = FP1~FP3 练习消耗/上交 6 套(锁定不可选)+ 正赛可用至多 7 套;
  排位套(Q1/Q2/Q3)带 6~7 圈磨损但可用于正赛;正赛必须至少使用 2 种干地配方(DP 约束 + 红旗提示)。
- **旗种口径**: SC 与 VSC 进站损失几乎相同,合并建模(通道 6s);红旗独立场景 —— 免费换胎
  (0 损失)+ 静态发车,换上更软/更新的胎在重启后前 N 圈获得抓地增益(默认 0.35s/圈 × 2 圈)。
- "完成超越圈数"定义: 成功样本中**最后一次落后之后的第 1 圈**(永久反超);0 = 无需追赶
  (自始至终领先,或经对手进站窗口直接获得位置)。
- 摩纳哥轮胎衰减极低,默认参数下 Undercut"对手不跟"等分支成功率接近 0 —— 这是模型对现实的正确反映;
  演示时可在"高级参数"中增大配方速度偏移差异(如 C5=-1.5、C3=+2.5)获得更直观的成功率对比。
- **AI 分析配置**(可选): 设置环境变量 `F1_AI_API_KEY`(及可选 `F1_AI_BASE_URL`/`F1_AI_MODEL`)
  后由大模型生成策略分析;未配置时使用本地规则分析,界面明确标注来源,不冒充 AI。
- 天气选项、2024 SC 历史验证未纳入本次前端(依赖任务④的历史回溯数据,留作扩展)。

## 八、Git 与仓库维护

- **远端**: <https://github.com/arrebolbay/F1-graduation-project>(main 分支)
- **日常同步**: 双击 `推送更新.bat`(自动 `git add -A` → `commit` → `push`),
  或命令行 `git add -A && git commit -m "说明" && git push origin main`。
- **不入库**(`.gitignore`): `缓存\`(FastF1 缓存,联网可重建)、`web\history.jsonl`、
  `__pycache__\`、`*.pyc`、运行日志等;其余数据(遥测 CSV、图表 PNG、结果表)全部纳入版本管理。
- **网络提示**: 若 `git push` 报 `Connection was reset`,多为 DNS/网络问题;
  本机已在 `~/.ssh/config` 将 `github.com` 固定到可用 GitHub 前端 IP(SSH 方式),
  失效时按配置文件内注释更换 IP,或启用系统代理后
  `git config --global http.proxy http://127.0.0.1:<端口>`(用完可 `--unset` 取消)。

## 九、License

本项目采用 [MIT License](LICENSE) 开源发布:

```
MIT License
Copyright (c) 2026 arrebolbay
```

自由使用、修改、分发(含商用),保留版权声明与许可文本即可。引用本项目数据/模型时,
建议注明数据来源为 FastF1(2022–2025 摩纳哥大奖赛正赛遥测)。
