import cv2
import numpy as np
import gradio as gr

# 初始化全局变量，存储控制点和目标点
points_src = []
points_dst = []
image = None

# 上传图像时清空控制点和目标点
def upload_image(img):
    global image, points_src, points_dst
    points_src.clear()  # 清空控制点
    points_dst.clear()  # 清空目标点
    image = img
    return img

# 记录点击点事件，并标记点在图像上，同时在成对的点间画箭头
def record_points(evt: gr.SelectData):
    global points_src, points_dst, image
    x, y = evt.index[0], evt.index[1]  # 获取点击的坐标
    
    # 判断奇偶次来分别记录控制点和目标点
    if len(points_src) == len(points_dst):
        points_src.append([x, y])  # 奇数次点击为控制点
    else:
        points_dst.append([x, y])  # 偶数次点击为目标点
    
    # 在图像上标记点（蓝色：控制点，红色：目标点），并画箭头
    marked_image = image.copy()
    for pt in points_src:
        cv2.circle(marked_image, tuple(pt), 1, (255, 0, 0), -1)  # 蓝色表示控制点
    for pt in points_dst:
        cv2.circle(marked_image, tuple(pt), 1, (0, 0, 255), -1)  # 红色表示目标点
    
    # 画出箭头，表示从控制点到目标点的映射
    for i in range(min(len(points_src), len(points_dst))):
        cv2.arrowedLine(marked_image, tuple(points_src[i]), tuple(points_dst[i]), (0, 255, 0), 1)  # 绿色箭头表示映射
    
    return marked_image

# 执行仿射变换

def point_guided_deformation(image, source_pts, target_pts, alpha=100.0, eps=1e-8):
    """ 
    Return
    ------
        A deformed image.
    """
    
    warped_image = np.array(image)
    image=np.array(warped_image)
    ### FILL: 基于MLS or RBF 实现 image warping
    n=len(source_pts)
  
    if(n):
        source_pts=np.array(source_pts)
        target_pts=np.array(target_pts)
        A=np.zeros((n,n),dtype=float)
        b=np.zeros((n,2),dtype=float)
        for i in range(n):
            for j in range(n):
                A[i,j]=1./((source_pts[i,0]-source_pts[j,0])**2+(source_pts[i,1]-source_pts[j,1])**2+alpha)
            b[i,0]=target_pts[i,0]-source_pts[i,0]
            b[i,1]=target_pts[i,1]-source_pts[i,1]
        
        coef=np.linalg.solve(A,b)
        A0=np.zeros(n,dtype=float)
        image_1=np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8) + np.array((255,255,255), dtype=np.uint8).reshape(1,1,3)
        for i in range(warped_image.shape[0]):
            for j in range(warped_image.shape[1]):
                for k in range(n):
                    A0[k]=1./((i-source_pts[k,1])**2+(j-source_pts[k,0])**2+alpha)
                result=np.array([0,0],dtype=int)
                result=np.round(np.dot(A0,coef))
                image_1[int(result[1]+i),int(result[0]+j)]=image[i,j]
        warped_image=image_1
        for i in range(warped_image.shape[0]):
            for j in range(warped_image.shape[1]):
                if(warped_image[i,j,0]==255 and warped_image[i,j,1]==255 and warped_image[i,j,2]==255):
                    sum1=0.
                    sum2=np.array([0,0,0],dtype=float)
                    for k in range(5):
                        for m in range(5):
                            if(warped_image[i-2+k,j-2+m,0]!=255 or warped_image[i-2+k,j-2+m,1]!=255 or warped_image[i-2+k,j-2+m,2]!=255):
                                sum1=sum1+1.
                                sum2=sum2+warped_image[i-2+k,j-2+m,:]
                               
                    sum2=np.ceil(sum2/sum1)
                    warped_image[i,j]=sum2
                    

        

    return warped_image

def run_warping():
    global points_src, points_dst, image ### fetch global variables

    warped_image = point_guided_deformation(image, np.array(points_src), np.array(points_dst))

    return warped_image

# 清除选中点
def clear_points():
    global points_src, points_dst
    points_src.clear()
    points_dst.clear()
    return image  # 返回未标记的原图

# 使用 Gradio 构建界面
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(sources="upload",label="上传图片", interactive=True, width=800, height=200)
            point_select = gr.Image(label="点击选择控制点和目标点", interactive=True, width=800, height=800)
            
        with gr.Column():
            result_image = gr.Image(label="变换结果", width=800, height=400)
    
    # 按钮
    run_button = gr.Button("Run Warping")
    clear_button = gr.Button("Clear Points")  # 添加清除按钮
    
    # 上传图像的交互
    input_image.upload(upload_image, input_image, point_select)
    # 选择点的交互，点选后刷新图像
    point_select.select(record_points, None, point_select)
    # 点击运行 warping 按钮，计算并显示变换后的图像
    run_button.click(run_warping, None, result_image)
    # 点击清除按钮，清空所有已选择的点
    clear_button.click(clear_points, None, point_select)
    
# 启动 Gradio 应用
demo.launch(share=True)
