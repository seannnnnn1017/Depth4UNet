import cv2
import numpy as np
import os
import glob
from tqdm import tqdm
import warnings
import time
warnings.filterwarnings('ignore')

class DepthEstimator:
    def __init__(self):
        """深度估計器"""
        self.current_image_name = ""
    
    def set_current_image(self, image_name):
        """設置當前處理的圖片名稱"""
        self.current_image_name = image_name
    
    def adaptive_preprocessing(self, image):
        """自適應預處理"""
        with tqdm(total=4, desc=f"[{self.current_image_name}] 自適應預處理", 
                 leave=False, ncols=100, ascii=True) as pbar:
            
            pbar.set_postfix_str("轉換灰階")
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            pbar.update(1)
            
            pbar.set_postfix_str("分析圖像特徵")
            contrast = np.std(gray)
            brightness = np.mean(gray)
            pbar.update(1)
            
            pbar.set_postfix_str("選擇處理策略")
            if contrast < 30:  # 低對比度
                pbar.set_postfix_str("低對比度增強")
                enhanced = self.enhance_low_contrast(image)
            elif contrast > 80:  # 高對比度
                pbar.set_postfix_str("高對比度平滑")
                enhanced = self.smooth_high_contrast(image)
            else:  # 正常對比度
                pbar.set_postfix_str("標準增強")
                enhanced = self.standard_enhancement(image)
            pbar.update(1)
            
            pbar.set_postfix_str("完成")
            pbar.update(1)
        
        return enhanced
    
    def enhance_low_contrast(self, image):
        """低對比度圖像增強"""
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l_channel = lab[:,:,0]
        
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
        l_channel = clahe.apply(l_channel)
        
        gamma = 0.8
        l_channel = np.power(l_channel/255.0, gamma) * 255
        l_channel = l_channel.astype(np.uint8)
        
        lab[:,:,0] = l_channel
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        return enhanced
    
    def smooth_high_contrast(self, image):
        """高對比度圖像平滑"""
        smoothed = cv2.bilateralFilter(image, 9, 75, 75)
        smoothed = cv2.GaussianBlur(smoothed, (3, 3), 0.5)
        return smoothed
    
    def standard_enhancement(self, image):
        """標準增強處理"""
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l_channel = lab[:,:,0]
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l_channel = clahe.apply(l_channel)
        
        lab[:,:,0] = l_channel
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        return enhanced
    
    def multi_scale_texture_analysis(self, image):
        """多尺度紋理分析"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        texture_maps = []
        scales = [3, 7, 15, 25]
        
        with tqdm(total=len(scales), desc=f"[{self.current_image_name}] 多尺度紋理分析", 
                 leave=False, ncols=100, ascii=True) as pbar:
            
            for scale in scales:
                pbar.set_postfix_str(f"處理尺度 {scale}")
                
                gabor_responses = []
                angles = np.arange(0, 180, 30)
                frequency = 0.1 + 0.1 * (scale / 25)
                
                # 內部角度處理進度
                for angle in angles:
                    kernel = cv2.getGaborKernel(
                        (scale*2+1, scale*2+1), scale/3, np.radians(angle), 
                        2*np.pi*frequency, 0.5, 0, ktype=cv2.CV_32F
                    )
                    filtered = cv2.filter2D(gray, cv2.CV_32F, kernel)
                    gabor_responses.append(np.abs(filtered))
                
                scale_texture = np.mean(gabor_responses, axis=0)
                texture_maps.append(scale_texture)
                pbar.update(1)
        
        # 加權融合
        with tqdm(total=2, desc=f"[{self.current_image_name}] 紋理融合", 
                 leave=False, ncols=100, ascii=True) as pbar:
            
            pbar.set_postfix_str("計算加權融合")
            weights = [0.4, 0.3, 0.2, 0.1]
            final_texture = np.zeros_like(texture_maps[0])
            
            for texture, weight in zip(texture_maps, weights):
                final_texture += texture * weight
            pbar.update(1)
            
            pbar.set_postfix_str("正規化結果")
            texture_depth = 255 - cv2.normalize(final_texture, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            pbar.update(1)
        
        return texture_depth
    
    def improved_shadow_analysis(self, image):
        """改進的陰影分析"""
        with tqdm(total=6, desc=f"[{self.current_image_name}] 陰影分析", 
                 leave=False, ncols=100, ascii=True) as pbar:
            
            pbar.set_postfix_str("轉換色彩空間")
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            pbar.update(1)
            
            pbar.set_postfix_str("HSV陰影檢測")
            h, s, v = cv2.split(hsv)
            shadow_hsv = (v < np.percentile(v, 25)) & (s < np.percentile(s, 40))
            pbar.update(1)
            
            pbar.set_postfix_str("LAB陰影檢測")
            l, a, b = cv2.split(lab)
            shadow_lab = l < np.percentile(l, 20)
            pbar.update(1)
            
            pbar.set_postfix_str("結合檢測結果")
            shadow_mask = shadow_hsv | shadow_lab
            pbar.update(1)
            
            pbar.set_postfix_str("形態學處理")
            kernel = np.ones((5,5), np.uint8)
            shadow_mask = cv2.morphologyEx(shadow_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
            shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_OPEN, kernel)
            pbar.update(1)
            
            pbar.set_postfix_str("距離變換")
            shadow_distance = cv2.distanceTransform(
                (~shadow_mask.astype(bool)).astype(np.uint8), 
                cv2.DIST_L2, 5
            )
            
            shadow_depth = cv2.normalize(shadow_distance, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            shadow_depth[shadow_mask.astype(bool)] = 200
            pbar.update(1)
        
        return shadow_depth
    
    def frequency_domain_analysis(self, image):
        """頻域分析"""
        with tqdm(total=5, desc=f"[{self.current_image_name}] 頻域分析", 
                 leave=False, ncols=100, ascii=True) as pbar:
            
            pbar.set_postfix_str("轉換灰階")
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            pbar.update(1)
            
            pbar.set_postfix_str("FFT變換")
            f_transform = np.fft.fft2(gray)
            f_shift = np.fft.fftshift(f_transform)
            pbar.update(1)
            
            pbar.set_postfix_str("創建高通濾波器")
            rows, cols = gray.shape
            crow, ccol = rows//2, cols//2
            
            mask = np.ones((rows, cols), np.uint8)
            r = 30
            center = [crow, ccol]
            x, y = np.ogrid[:rows, :cols]
            mask_area = (x - center[0])**2 + (y - center[1])**2 <= r*r
            mask[mask_area] = 0
            pbar.update(1)
            
            pbar.set_postfix_str("應用濾波器")
            f_shift_filtered = f_shift * mask
            pbar.update(1)
            
            pbar.set_postfix_str("反變換和正規化")
            f_ishift = np.fft.ifftshift(f_shift_filtered)
            img_back = np.fft.ifft2(f_ishift)
            img_back = np.abs(img_back)
            
            frequency_depth = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            pbar.update(1)
        
        return frequency_depth
    
    def gradient_coherence_analysis(self, image):
        """梯度一致性分析"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
        h, w = gray.shape
        
        with tqdm(total=4, desc=f"[{self.current_image_name}] 梯度分析", 
                 leave=False, ncols=100, ascii=True) as pbar:
            
            pbar.set_postfix_str("計算梯度")
            grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            
            magnitude = np.sqrt(grad_x**2 + grad_y**2)
            direction = np.arctan2(grad_y, grad_x)
            pbar.update(1)
            
            pbar.set_postfix_str("初始化一致性圖")
            kernel_size = 15
            consistency_map = np.zeros_like(magnitude)
            pbar.update(1)
            
            pbar.set_postfix_str("計算局部一致性")
            total_pixels = (h - kernel_size) * (w - kernel_size)
            
            # 使用內部進度條顯示像素處理進度
            pixel_pbar = tqdm(total=total_pixels, desc="      處理像素", 
                            leave=False, ncols=80, ascii=True)
            
            for i in range(kernel_size//2, h - kernel_size//2):
                for j in range(kernel_size//2, w - kernel_size//2):
                    local_dir = direction[i-kernel_size//2:i+kernel_size//2+1, 
                                       j-kernel_size//2:j+kernel_size//2+1]
                    local_mag = magnitude[i-kernel_size//2:i+kernel_size//2+1, 
                                        j-kernel_size//2:j+kernel_size//2+1]
                    
                    if np.sum(local_mag) > 0:
                        weighted_cos = np.sum(local_mag * np.cos(local_dir - direction[i,j]))
                        weighted_sin = np.sum(local_mag * np.sin(local_dir - direction[i,j]))
                        coherence = np.sqrt(weighted_cos**2 + weighted_sin**2) / np.sum(local_mag)
                        consistency_map[i, j] = coherence
                    
                    pixel_pbar.update(1)
            
            pixel_pbar.close()
            pbar.update(1)
            
            pbar.set_postfix_str("正規化結果")
            coherence_depth = 255 - cv2.normalize(consistency_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            pbar.update(1)
        
        return coherence_depth
    
    def advanced_edge_preserving_fusion(self, depth_maps, weights, reference_image):
        """高級邊緣保持融合"""
        with tqdm(total=4, desc=f"[{self.current_image_name}] 邊緣保持融合", 
                 leave=False, ncols=100, ascii=True) as pbar:
            
            pbar.set_postfix_str("檢測邊緣")
            gray = cv2.cvtColor(reference_image, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 30, 100)
            edges = cv2.dilate(edges, np.ones((3,3)), iterations=1)
            pbar.update(1)
            
            pbar.set_postfix_str("計算邊緣權重")
            edge_weights = []
            for depth_map in depth_maps:
                grad_x = cv2.Sobel(depth_map, cv2.CV_32F, 1, 0, ksize=3)
                grad_y = cv2.Sobel(depth_map, cv2.CV_32F, 0, 1, ksize=3)
                depth_gradient = np.sqrt(grad_x**2 + grad_y**2)
                
                edge_reliability = depth_gradient * (edges / 255.0)
                edge_weights.append(edge_reliability)
            pbar.update(1)
            
            pbar.set_postfix_str("正規化權重")
            total_edge_weight = np.sum(edge_weights, axis=0)
            total_edge_weight[total_edge_weight == 0] = 1
            
            edge_weights = [ew / total_edge_weight for ew in edge_weights]
            pbar.update(1)
            
            pbar.set_postfix_str("融合深度圖")
            fused_depth = np.zeros_like(depth_maps[0], dtype=np.float32)
            
            for i, (depth_map, weight) in enumerate(zip(depth_maps, weights)):
                dynamic_weight = weight * 0.5 + edge_weights[i] * 0.5
                fused_depth += depth_map.astype(np.float32) * dynamic_weight
            pbar.update(1)
        
        return fused_depth.astype(np.uint8)
    
    def post_processing_pipeline(self, depth_map, reference_image):
        """後處理管道"""
        with tqdm(total=4, desc=f"[{self.current_image_name}] 後處理", 
                 leave=False, ncols=100, ascii=True) as pbar:
            
            pbar.set_postfix_str("雙邊濾波去噪")
            denoised = cv2.bilateralFilter(depth_map, 9, 75, 75)
            pbar.update(1)
            
            pbar.set_postfix_str("邊緣檢測")
            gray = cv2.cvtColor(reference_image, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edge_mask = edges / 255.0
            pbar.update(1)
            
            pbar.set_postfix_str("邊緣增強")
            enhanced = denoised * (1 - edge_mask) + depth_map * edge_mask
            pbar.update(1)
            
            pbar.set_postfix_str("最終平滑")
            smoothed = cv2.GaussianBlur(enhanced.astype(np.uint8), (3, 3), 0.5)
            final = enhanced * edge_mask + smoothed * (1 - edge_mask)
            pbar.update(1)
        
        return final.astype(np.uint8)
    
    def estimate_depth(self, image_rgb):
        """對單張圖片進行深度估計"""
        print(f"/n🔍 開始處理: {self.current_image_name}")
        
        # 總進度追蹤
        with tqdm(total=6, desc=f"[{self.current_image_name}] 總體進度", 
                 position=0, leave=True, ncols=120, ascii=True) as main_pbar:
            
            # 1. 自適應預處理
            main_pbar.set_postfix_str("自適應預處理")
            enhanced_image = self.adaptive_preprocessing(image_rgb)
            main_pbar.update(1)
            
            # 2. 多尺度紋理分析
            main_pbar.set_postfix_str("多尺度紋理分析")
            texture_depth = self.multi_scale_texture_analysis(enhanced_image)
            main_pbar.update(1)
            
            # 3. 陰影分析
            main_pbar.set_postfix_str("陰影分析")
            shadow_depth = self.improved_shadow_analysis(enhanced_image)
            main_pbar.update(1)
            
            # 4. 頻域分析
            main_pbar.set_postfix_str("頻域分析")
            frequency_depth = self.frequency_domain_analysis(enhanced_image)
            main_pbar.update(1)
            
            # 5. 梯度一致性分析
            main_pbar.set_postfix_str("梯度一致性分析")
            coherence_depth = self.gradient_coherence_analysis(enhanced_image)
            main_pbar.update(1)
            
            # 6. 融合和後處理
            main_pbar.set_postfix_str("融合和後處理")
            depth_maps = [texture_depth, shadow_depth, frequency_depth, coherence_depth]
            weights = [0.35, 0.25, 0.25, 0.15]
            
            fused_depth = self.advanced_edge_preserving_fusion(depth_maps, weights, enhanced_image)
            final_depth = self.post_processing_pipeline(fused_depth, enhanced_image)
            main_pbar.update(1)
            
            main_pbar.set_postfix_str("完成")
        
        return final_depth

def process_folder(input_folder, output_folder):
    """批次處理資料夾中的所有圖片"""
    
    # 創建輸出資料夾
    os.makedirs(output_folder, exist_ok=True)
    
    # 支援的圖片格式
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
    
    # 收集所有圖片檔案（避免重複）
    image_files = set()  # 使用set避免重複
    for extension in image_extensions:
        image_files.update(glob.glob(os.path.join(input_folder, extension)))
        image_files.update(glob.glob(os.path.join(input_folder, extension.upper())))
    
    image_files = sorted(list(image_files))  # 轉回list並排序
    
    if not image_files:
        print(f"❌ 在資料夾 {input_folder} 中沒有找到圖片檔案")
        return
    
    print("="*80)
    print(f"🚀 開始批次深度估計處理")
    print("="*80)
    print(f"📁 輸入資料夾: {input_folder}")
    print(f"📤 輸出資料夾: {output_folder}")
    print(f"📊 找到 {len(image_files)} 張圖片")
    print("="*80)
    
    # 初始化深度估計器
    estimator = DepthEstimator()
    
    successful = 0
    failed = 0
    start_time = time.time()
    
    # 處理每張圖片
    for i, image_path in enumerate(image_files, 1):
        try:
            filename = os.path.basename(image_path)
            print(f"/n📷 [{i}/{len(image_files)}] {filename}")
            
            # 設置當前處理的圖片名稱
            estimator.set_current_image(filename)
            
            # 讀取圖片
            image = cv2.imread(image_path)
            if image is None:
                print(f"❌ 無法讀取: {image_path}")
                failed += 1
                continue
            
            # 轉換為RGB格式
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 進行深度估計
            depth_map = estimator.estimate_depth(image_rgb)
            
            # 生成輸出檔名（保持原始檔名）
            name, ext = os.path.splitext(filename)
            output_path = os.path.join(output_folder, f"{name}_depth{ext}")
            
            # 保存深度圖
            cv2.imwrite(output_path, depth_map)
            successful += 1
            
            print(f"✅ 完成: {filename} → {name}_depth{ext}")
            
            # 計算預估剩餘時間
            elapsed_time = time.time() - start_time
            if i > 0:
                avg_time_per_image = elapsed_time / i
                remaining_images = len(image_files) - i
                eta = avg_time_per_image * remaining_images
                eta_str = f"{eta//60:.0f}分{eta%60:.0f}秒" if eta >= 60 else f"{eta:.0f}秒"
                print(f"⏱️  預估剩餘時間: {eta_str}")
            
        except Exception as e:
            print(f"❌ 處理失敗 {filename}: {e}")
            failed += 1
    
    # 完成報告
    total_time = time.time() - start_time
    total_time_str = f"{total_time//60:.0f}分{total_time%60:.0f}秒" if total_time >= 60 else f"{total_time:.0f}秒"
    
    print("/n" + "="*80)
    print(f"🎉 批次處理完成！")
    print("="*80)
    print(f"⏱️  總處理時間: {total_time_str}")
    print(f"✅ 成功處理: {successful} 張")
    print(f"❌ 處理失敗: {failed} 張")
    print(f"📈 成功率: {(successful/(successful+failed)*100):.1f}%" if (successful+failed) > 0 else "0%")
    print(f"📁 結果保存在: {output_folder}")
    print("="*80)

def main():
    """主函數"""
    # 設定輸入和輸出資料夾路徑
    input_folder = r"AerialImageDataset\train\images"      # 修改為你的輸入資料夾
    output_folder = r"depth/output"     # 修改為你的輸出資料夾
    
    # 檢查輸入資料夾是否存在
    if not os.path.exists(input_folder):
        print(f"❌ 輸入資料夾不存在: {input_folder}")
        print("請修改 input_folder 路徑為你的實際圖片資料夾")
        return
    
    # 開始批次處理
    process_folder(input_folder, output_folder)

if __name__ == "__main__":
    main()