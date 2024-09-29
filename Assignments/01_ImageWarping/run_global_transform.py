import gradio as gr
import cv2
import numpy as np

# Function to convert 2x3 affine matrix to 3x3 for matrix multiplication
def to_3x3(affine_matrix):
    return np.vstack([affine_matrix, [0, 0, 1]])

# Function to apply transformations based on user inputs
def apply_transform(image, scale, rotation, translation_x, translation_y, flip_horizontal):

    # Convert the image from PIL format to a NumPy array
    image = np.array(image)
    # Pad the image to avoid boundary issues
    pad_size = min(image.shape[0], image.shape[1]) // 2
    image_new = np.zeros((pad_size*2+image.shape[0], pad_size*2+image.shape[1], 3), dtype=np.uint8) + np.array((255,255,255), dtype=np.uint8).reshape(1,1,3)
    image_new[pad_size:pad_size+image.shape[0], pad_size:pad_size+image.shape[1]] = image
    image = np.array(image_new)
    transformed_image = np.array(image)

    ### FILL: Apply Composition Transform 
    # Note: for scale and rotation, implement them around the center of the image （围绕图像中心进行放缩和旋转）
    

    if flip_horizontal:
        M_f=np.array([[-1,0,transformed_image.shape[1]],[0,1,0]],dtype=np.float32)
        transformed_image = cv2.warpAffine(transformed_image, M_f, (transformed_image.shape[1], transformed_image.shape[0]))
    '''
    Y,X=np.meshgrid(np.arange(transformed_image.shape[0]),np.arange(transformed_image.shape[1]),indexing="ij")
    z= np.column_stack((Y.ravel(), X.ravel()))
    '''
    pi=np.pi
    theta=rotation/180.*pi
    cos_theta=np.cos(theta)
    sin_theta=np.sin(theta)
    delta_1=(1-cos_theta)*transformed_image.shape[1]//2+sin_theta*transformed_image.shape[0]//2+translation_x
    delta_2=(-sin_theta)*transformed_image.shape[1]//2+(1-cos_theta)*transformed_image.shape[0]//2+translation_y
    '''
    z_new=np.zeros((transformed_image.shape[0]*transformed_image.shape[1],2),dtype=int)
    z_new[:,1]=np.round(z[:,1]*cos_theta-sin_theta*z[:,0]+delta_1)
    z_new[:,0]=np.round(z[:,1]*cos_theta+sin_theta*z[:,0]+delta_2)
    image_new1=transformed_image.reshape(transformed_image.shape[0]*transformed_image.shape[1],3)
    image_new2=np.zeros(((image.shape[0])* (image.shape[1]), 3), dtype=np.uint8) + np.array((255,255,255), dtype=np.uint8).reshape(1,3)
    flag=np.logical_and.reduce([z_new[:,0]>=0,z_new[:,0]<transformed_image.shape[0]-1,z_new[:,1]>=0,z_new[:,1]<transformed_image.shape[1]-1])
    print(z_new[flag].shape)
    image_new2[z_new[flag,0]*transformed_image.shape[1]+z_new[flag,1]]=image_new1[flag]
    transformed_image=image_new2.reshape(transformed_image.shape[0],transformed_image.shape[1],3)
    '''
    
    M=np.array([[np.cos(theta),-np.sin(theta),translation_x+(1-np.cos(theta))*transformed_image.shape[1]//2+np.sin(theta)*transformed_image.shape[0]//2],[np.sin(theta),np.cos(theta),translation_y+(-np.sin(theta))*transformed_image.shape[1]//2+(1-np.cos(theta))*transformed_image.shape[0]//2]],dtype=np.float32)
    image_new2=np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8) + np.array((255,255,255), dtype=np.uint8).reshape(1,1,3)
    for i in range(transformed_image.shape[1]):
        for j in range(transformed_image.shape[0]):
            new_x=cos_theta*i-sin_theta*j+delta_1
            new_y=sin_theta*i+cos_theta*j+delta_2
            if new_x>=transformed_image.shape[1]-1 or new_y>=transformed_image.shape[0]-1:
                continue
            new_x=int(new_x)
            new_y=int(new_y)
            image_new2[new_y,new_x]=transformed_image[j,i]
    transformed_image=np.array(image_new2)
    '''
    旋转和放大要对矩阵进行填充，效果不佳。
    '''
    transformed_image=cv2.warpAffine(image, M, (transformed_image.shape[1], transformed_image.shape[0]))
    M_scale = np.array([[scale, 0, (1-scale)*(transformed_image.shape[1]//2)], [0, scale, (1-scale)*(transformed_image.shape[0]//2)]],dtype=np.float32)
    transformed_image = cv2.warpAffine(transformed_image, M_scale, (transformed_image.shape[1], transformed_image.shape[0]))
    return transformed_image

# Gradio Interface
def interactive_transform():
    with gr.Blocks() as demo:
        gr.Markdown("## Image Transformation Playground")
        
        # Define the layout
        with gr.Row():
            # Left: Image input and sliders
            with gr.Column():
                image_input = gr.Image(type="pil", label="Upload Image")

                scale = gr.Slider(minimum=0.1, maximum=2.0, step=0.1, value=1.0, label="Scale")
                rotation = gr.Slider(minimum=-180, maximum=180, step=1, value=0, label="Rotation (degrees)")
                translation_x = gr.Slider(minimum=-300, maximum=300, step=10, value=0, label="Translation X")
                translation_y = gr.Slider(minimum=-300, maximum=300, step=10, value=0, label="Translation Y")
                flip_horizontal = gr.Checkbox(label="Flip Horizontal")
            
            # Right: Output image
            image_output = gr.Image(label="Transformed Image")
        
        # Automatically update the output when any slider or checkbox is changed
        inputs = [
            image_input, scale, rotation, 
            translation_x, translation_y, 
            flip_horizontal
        ]

        # Link inputs to the transformation function
        image_input.change(apply_transform, inputs, image_output)
        scale.change(apply_transform, inputs, image_output)
        rotation.change(apply_transform, inputs, image_output)
        translation_x.change(apply_transform, inputs, image_output)
        translation_y.change(apply_transform, inputs, image_output)
        flip_horizontal.change(apply_transform, inputs, image_output)

    return demo

# Launch the Gradio interface
interactive_transform().launch(share=True)
