视觉定位（Visual Grounding，后简称 Grounding）是根据对自然语言的任务要求，在图片中找到对应的目标，并返回目标的坐标，需要模型具备视觉理解以及自然语言理解的能力。
和目标检测最大的区别在于，输入多了自然语言信息，在对物体进行定位时需要理解自然语言的信息。使用上更加灵活交互更加自然。

<span aceTableMode="list" aceTableWidth="3,3,10"></span>
|**输入** |**效果预览** |‌ |
|---|---|---|
|<span>![图片](https://p9-arcosite.byteimg.com/tos-cn-i-goo7wpa0wc/f9a8b28c25ac4d86979de8f52e73def6~tplv-goo7wpa0wc-image.image =1024x) </span>|<span>![图片](https://p9-arcosite.byteimg.com/tos-cn-i-goo7wpa0wc/114267ea04e7424a8dc2e168c55b2ae8~tplv-goo7wpa0wc-image.image =1024x) </span>|‌ |\
|> 提示词：框出中间狼卡通形象的头部的位置，输出 bounding box 的坐标 |> 模型回答：`<bbox>175 98 791 476</bbox>` | |

<span id="90aef7dd"></span>
# 支持模型
请参见[视觉理解能力](/docs/82379/1330310#ff5ef604)。
<span id="750b19d5"></span>
# 使用流程
下面将基于示例代码，详细介绍 Grounding 的使用流程。
:::tip
方舟平台的新用户？获取 API Key 及 开通模型等准备工作，请参见 [快速入门](/docs/82379/1399008)。
:::
<span id="cb799c3b"></span>
## 1. 参数配置
首先导入必要的库，设置模型ID、图片路径和提示词等参数，并从环境变量中读取API密钥。
```Python
import os
import base64
import cv2
from volcenginesdkarkruntime import Ark

# 配置参数
DEFAULT_MODEL = "doubao-seed-2-0-lite-260215"  # Replace with Model ID .
IMAGE_PATH = "./ark_demo_img.png"
PROMPT = "框出中间狼卡通形象的头部的位置，输出 bounding box 的坐标"
BBOX_TAG_START = "<bbox>"
BBOX_TAG_END = "</bbox>"

# 读取API密钥
api_key = os.getenv("ARK_API_KEY")

# 创建Ark客户端
client = Ark(
    api_key=api_key,
    # The base URL for model invocation
    base_url="https://ark.cn-beijing.volces.com/api/v3", 
)
```

<span id="c6651960"></span>
## 2. 图片读取与Base64编码
读取本地图片文件，并将其转换为Base64字符串格式，以便通过API传输。
```Python
with open(IMAGE_PATH, "rb") as f:
    base64_image = base64.b64encode(f.read()).decode('utf-8')
```

<span id="10dc2486"></span>
## 3. 调用API生成边界框
将Base64编码的图片和文本提示词作为多模态输入，发送给grounding模型获取预测结果。
```Python
response = client.chat.completions.create(
    model=DEFAULT_MODEL,
    messages=[{
        "role": "user",
        "content": [{
            "type": "image_url",  # 图片输入
            "image_url": {"url": f"data:image/png;base64,{base64_image}"}
        }, {
            "type": "text",  # 文本提示
            "text": PROMPT
        }]
    }]
)
bbox_content = response.choices[0].message.content
```

<span id="cbfeb48d"></span>
## 4. 解析模型返回的边界框坐标
从模型返回的结果中提取边界框坐标，并验证其格式是否符合预期。
坐标格式为 `<bbox>x_min y_min x_max y_max</bbox>`，其中(`x_min`, `y_min`)为方框左上角的坐标，(`x_max`, `y_max` )为方框右下角的坐标。
```Python
# 检查结果格式是否正确
if not (bbox_content.startswith(BBOX_TAG_START) and bbox_content.endswith(BBOX_TAG_END)):
    print("错误：边界框格式不正确，缺少标签包裹")
    exit(1)

# 解析坐标值
try:
    coords_str = bbox_content[len(BBOX_TAG_START):-len(BBOX_TAG_END)]
    coords = list(map(int, coords_str.split()))
    if len(coords) != 4:  # 验证坐标数量(xmin, ymin, xmax, ymax)
        raise ValueError("坐标数量不正确，需要4个数值")
    x_min, y_min, x_max, y_max = coords
except ValueError as e:
    print(f"坐标解析失败: {str(e)}")
    exit(1)
```

<span id="7de20f90"></span>
## 5. 在原图上绘制边界框并保存
转化坐标，`x_min`, `y_min`, `x_max`, `y_max`是 **归一化到 1000\*1000 的比例坐标**，**取值范围 [0, 999]** 。图片宽高等分为1000后，图片左上角为原点绘制坐标系，对应点的坐标位置，原理示意如下图所示。
<span>![图片](https://p9-arcosite.byteimg.com/tos-cn-i-goo7wpa0wc/e247476bb66b4b4f945b48318c18ee9b~tplv-goo7wpa0wc-image.image =200x) </span>
根据图像实际尺寸缩放边界框坐标，将模型输出的相对坐标映射到绝对坐标，需进行计算，例如左上角坐标横轴绝对值是 `x=x_min/1000*w`，w 是图片的实际宽度。
```Python
# 读取原图
image = cv2.imread(IMAGE_PATH)

# 获取图像尺寸并缩放坐标
h, w = image.shape[:2]
x_min_real = int(x_min * w / 1000)
y_min_real = int(y_min * h / 1000)
x_max_real = int(x_max * w / 1000)
y_max_real = int(y_max * h / 1000)

# 绘制红色边界框
cv2.rectangle(image, (x_min_real, y_min_real), (x_max_real, y_max_real), (0, 0, 255), 3)

# 保存结果图片
output_path = os.path.splitext(IMAGE_PATH)[0] + "_with_bboxes.png"
cv2.imwrite(output_path, image)
print(f"成功保存标注图片: {output_path}")
```

<span id="4c0fb035"></span>
## 结果预览
配置并运行该示例代码，在代码路径下生成绘制后的图片预览。
<span>![图片](https://p9-arcosite.byteimg.com/tos-cn-i-goo7wpa0wc/114267ea04e7424a8dc2e168c55b2ae8~tplv-goo7wpa0wc-image.image =163x) </span>
<span id="93a6289f"></span>
# 使用说明

* 不建议将坐标结果（ bounding box 坐标）放在 json 中。
* 不建议指定 bounding box 的输出结构。

<span id="aa6f318c"></span>
# 更多示例

<span aceTableMode="list" aceTableWidth="3,3,4,3"></span>
|场景及说明 |Prompt关键词 |Prompt示例 |效果图 |
|---|---|---|---|
|带属性物体检测|`<bbox>x1 y1 x2 y2</bbox>` |`帮我框选森林里的着火范围，以<bbox>x1 y1 x2 y2</bbox>的形式表示` |<span>![图片](https://p9-arcosite.byteimg.com/tos-cn-i-goo7wpa0wc/e8140cc3d68240fcaab0a92be13101bd~tplv-goo7wpa0wc-image.image =1151x) </span> |\
|> 需要定位图像中符合自然语言描述的特定属性的对象 | | | |
|多目标检测|`[{"category": category, "bbox": "<bbox>x1 y1 x2 y2</bbox>"}, {"category": category, "bbox": "<bbox>x1 y1 x2 y2</bbox>"}]` |请检测图像中所有属于 "plate（盘子）、photo（照片）、kid（小孩）、cup（杯子）" 类别的物体。对于每个物体，请提供其类别和边界框，格式为：`[{"category": category, "bbox": "<bbox>x1 y1 x2 y2</bbox>"}, {"category": category, "bbox": "<bbox>x1 y1 x2 y2</bbox>"}]` |<span>![图片](https://p9-arcosite.byteimg.com/tos-cn-i-goo7wpa0wc/f768fcf0037a4baf80b2d4157898d8a2~tplv-goo7wpa0wc-image.image =1280x) </span> |\
|> 需要同时检测多个预定义类别的对象 | | | |
|根据图像信息定位目标|`<bbox>x1 y1 x2 y2</bbox>` |请根据第一张图的主要目标，识别第二张图中类似的目标，对于每个物体提供其边界框，格式为：`<bbox>x1 y1 x2 y2</bbox>` |<span>![图片](https://p9-arcosite.byteimg.com/tos-cn-i-goo7wpa0wc/8ca3750ecfd540f285703e11f3c9edf9~tplv-goo7wpa0wc-image.image =1024x) </span> |\
|> 根据目标的图像信息在另一张图中进行定位 | | | |
|计算物体数量|`<point>x y</point>` |定位画面中水面上的所有人，输出 point，格式`<point>x1 y1</point>`，并计数水面中有多少人？ |<span>![图片](https://p9-arcosite.byteimg.com/tos-cn-i-goo7wpa0wc/2392404be5034344b31fc99b9e637dac~tplv-goo7wpa0wc-image.image =1151x) </span> |\
|> 需要统计特定对象的数量 | | | |
|识别图像文本|`<text>text</text><polygon>x1 y1, x2 y2, x3 y3, x4 y4</polygon>` |标注图中的文字，格式为`<text>text</text><polygon>x1 y1, x2 y2, x3 y3, x4 y4</polygon>` |<span>![图片](https://p9-arcosite.byteimg.com/tos-cn-i-goo7wpa0wc/7ef81d129cb34b5a8e3e1c8cc5d67f7e~tplv-goo7wpa0wc-image.image =1280x) </span> |\
|> 需要提取图像中的文字内容及位置 | | | |
|3D物体检测|`<图像相机的详细参数>`|以下是该图像的详细相机参数。|<span>![图片](https://p9-arcosite.byteimg.com/tos-cn-i-goo7wpa0wc/b8c5fa94a87c4977874c345630e0179e~tplv-goo7wpa0wc-image.image =1280x) </span> |\
|> 检测图像中的3D立体格式的物体 |`<3dbbox>x_center y_center z_center x_size y_size z_size pitch yaw roll</3dbbox>` |`相机内参：焦距f_x=5545.08, f_y=5545.08。主点坐标位于图像中心附近，因此当图像宽度为4284，高度为5712时，c_x=2142.00且c_y=2856.00。这里我们不考虑畸变参数。因此，内参矩阵K = [[1460.00, 0, 2142.00], [0, 1460.00, 2856.00], [0, 0, 1]]。`| |\
| | |`相机坐标系：X 轴向右，Y 轴向下，Z 轴向前。原点为相机位置。我们将相机坐标系作为世界坐标系，即相机外参矩阵为[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]。`| |\
| | |请按照以下格式输出每个3D框：`<3dbbox>x_center y_center z_center x_size y_size z_size pitch yaw roll</3dbbox>`| |\
| | |注意：| |\
| | |(1) x_center, y_center, z_center：目标中心在相机坐标系中沿着XYZ轴的位置，单位为米。| |\
| | |(2) x_size, y_size, z_size：当旋转角为零时，目标沿XYZ轴的尺寸，单位为米。| |\
| | |(3) pitch, yaw, roll：分别为围绕XYZ轴的欧拉角。这里的每个角度值被归一化到（\-1,1），需要乘以180转换到实际的角度。在这幅图中检测杯子，并以3D边界框格式展示结果。检测图像中的杯子，并以 3D框形式显示结果 | |