import base64
import time
import requests
import os
import uuid
from typing import Dict, List, Optional, Any
from io import BytesIO
from PIL import Image
import numpy as np
from dashscope import MultiModalConversation
import dashscope

try:
    from .config import (
        DASHSCOPE_API_KEY, OPENAI_API_KEY, SERPER_API_KEY,
        DASHSCOPE_BASE_URL, OPENAI_BASE_URL, DASHSCOPE_IMAGE_EDIT_URL,
        MAX_RETRIES, BASE_DELAY, MAX_DELAY, REQUEST_TIMEOUT
    )
except ImportError:
    from config import (
        DASHSCOPE_API_KEY, OPENAI_API_KEY, SERPER_API_KEY,
        DASHSCOPE_BASE_URL, OPENAI_BASE_URL, DASHSCOPE_IMAGE_EDIT_URL,
        MAX_RETRIES, BASE_DELAY, MAX_DELAY, REQUEST_TIMEOUT
    )

TMP_IMAGE_BASE_DIR = path_to_tmp-image

import os
_tmp_suffix = os.environ.get("TMP_IMAGE_SUFFIX", "")
if _tmp_suffix:
    TMP_IMAGE_DIR = os.path.join(TMP_IMAGE_BASE_DIR, _tmp_suffix)
else:
    TMP_IMAGE_DIR = TMP_IMAGE_BASE_DIR

os.makedirs(TMP_IMAGE_DIR, exist_ok=True)


class O3Client:

    def __init__(self, model_name: str = "o3-2025-04-16"):
        self.model_name = model_name
        
        if self._is_qwen_model(model_name):
            self.api_key = DASHSCOPE_API_KEY
            self.base_url = DASHSCOPE_BASE_URL
            self.api_provider = "dashscope"
        else:
            self.api_key = OPENAI_API_KEY
            self.base_url = OPENAI_BASE_URL
            self.api_provider = "openai"
        
        if not self.api_key:
            print(f"⚠️  警告: {self.api_provider.upper()} API Key 未设置 (模型: {model_name})")
    
    @staticmethod
    def _is_qwen_model(model_name: str) -> bool:
        qwen_keywords = ['qwen', 'qwen-vl', 'qwen3-vl', 'qwen2-vl']
        model_lower = model_name.lower()
        return any(keyword in model_lower for keyword in qwen_keywords)
    
    def _compress_image_if_needed(self, image: Image.Image, max_size_mb: float = 4.5) -> Image.Image:

        max_size_bytes = int(max_size_mb * 1024 * 1024)
        
        buffer = BytesIO()
        image.save(buffer, format='JPEG', quality=95)
        current_size = buffer.tell()
        
        if current_size <= max_size_bytes:
            return image
        
        print(f"[O3Client] ⚠️  图片过大 ({current_size / 1024 / 1024:.2f}MB)，开始压缩...")
        
        original_width, original_height = image.size
        scale_factor = (max_size_bytes / current_size) ** 0.5
        
        scale_factor = max(0.3, min(0.7, scale_factor))
        
        new_width = int(original_width * scale_factor)
        new_height = int(original_height * scale_factor)
        
        compressed_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        for quality in [85, 75, 65, 55, 45]:
            buffer = BytesIO()
            compressed_image.save(buffer, format='JPEG', quality=quality)
            final_size = buffer.tell()
            
            if final_size <= max_size_bytes:
                print(f"[O3Client] ✓ 压缩成功: {original_width}x{original_height} -> {new_width}x{new_height}, "
                      f"{current_size / 1024 / 1024:.2f}MB -> {final_size / 1024 / 1024:.2f}MB (质量={quality})")
                return compressed_image
        
        print(f"[O3Client] ⚠️  继续缩小分辨率...")
        scale_factor = 0.5
        new_width = int(new_width * scale_factor)
        new_height = int(new_height * scale_factor)
        compressed_image = compressed_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        buffer = BytesIO()
        compressed_image.save(buffer, format='JPEG', quality=75)
        final_size = buffer.tell()
        
        print(f"[O3Client] ✓ 最终压缩: {original_width}x{original_height} -> {new_width}x{new_height}, "
              f"{current_size / 1024 / 1024:.2f}MB -> {final_size / 1024 / 1024:.2f}MB")
        
        return compressed_image
    
    def _is_base64_string(self, s: str) -> bool:

        if not s or len(s) < 100:
            return False
        
        if s.startswith('/') or s.startswith('./') or s.startswith('../') or '\\' in s:
            return False
        
        if s.startswith('iVBORw0KGgo') or s.startswith('/9j/') or s.startswith('R0lGODlh'):
            return True
        
        base64_chars = set('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=')
        return all(c in base64_chars for c in s[:100])
    
    def encode_image_to_base64(self, image_path: str) -> Optional[str]:
        try:
            if self._is_base64_string(image_path):
                print(f"[O3Client] 检测到已是 base64 图片，直接返回")
                return image_path
            
            image = Image.open(image_path)
            
            if image.mode not in ('RGB', 'L'):
                image = image.convert('RGB')
            
            compressed_image = self._compress_image_if_needed(image)
            
            buffer = BytesIO()
            compressed_image.save(buffer, format='JPEG', quality=95)
            img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            return img_str
        except Exception as e:
            print(f"图片编码失败 {image_path[:100] if len(image_path) > 100 else image_path}")
            return None
    
    def encode_pil_image_to_base64(self, image: Image.Image) -> str:
        if image.mode not in ('RGB', 'L'):
            image = image.convert('RGB')
        
        compressed_image = self._compress_image_if_needed(image)
        
        buffer = BytesIO()
        compressed_image.save(buffer, format='JPEG', quality=95)
        img_str = base64.b64encode(buffer.getvalue()).decode()
        
        return img_str
    
    def chat(self, messages: List[Dict[str, Any]], image_path: Optional[str] = None,
             image_pil: Optional[Image.Image] = None, image_url: Optional[str] = None,
             image_urls: Optional[List[str]] = None,  # 新增：支持多图片
             max_tokens: int = 8196) -> Optional[str]:

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"{self.api_key}"
        }
        
        if image_urls and len(image_urls) > 0:
            if messages and messages[-1]["role"] == "user":
                content = messages[-1]["content"]
                content_array = []
                
                print(f"[O3Client] 处理 {len(image_urls)} 张图片...")
                for idx, img_url in enumerate(image_urls, 1):
                    if img_url.startswith("file://"):
                        local_path = img_url.replace("file://", "")
                        print(f"  [{idx}/{len(image_urls)}] 编码本地图片: {local_path[-50:]}...")
                        base64_image = self.encode_image_to_base64(local_path)
                        if base64_image:
                            print(f"  [{idx}/{len(image_urls)}] ✓ 编码完成 (大小: {len(base64_image)//1024}KB)")
                            content_array.append({
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            })
                        else:
                            print(f"  [{idx}/{len(image_urls)}] ✗ 编码失败")
                    else:
                        print(f"  [{idx}/{len(image_urls)}] 使用网络URL: {img_url[:60]}...")
                        content_array.append({
                            "type": "image_url",
                            "image_url": {
                                "url": img_url
                            }
                        })
                
                print(f"[O3Client] 图片处理完成，准备发送API请求...")
                
                content_array.append({
                    "type": "text",
                    "text": content
                })
                
                messages[-1]["content"] = content_array
        elif image_url and image_url.strip():
            if image_url.startswith("file://"):
                local_path = image_url.replace("file://", "")
                base64_image = self.encode_image_to_base64(local_path)
                if base64_image and messages and messages[-1]["role"] == "user":
                    content = messages[-1]["content"]
                    messages[-1]["content"] = [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        },
                        {
                            "type": "text",
                            "text": content
                        }
                    ]
            else:
                if messages and messages[-1]["role"] == "user":
                    content = messages[-1]["content"]
                    messages[-1]["content"] = [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_url
                            }
                        },
                        {
                            "type": "text",
                            "text": content
                        }
                    ]
        elif image_path and image_path.strip():
            base64_image = self.encode_image_to_base64(image_path)
            if base64_image and messages and messages[-1]["role"] == "user":
                content = messages[-1]["content"]
                messages[-1]["content"] = [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    },
                    {
                        "type": "text",
                        "text": content
                    }
                ]
        elif image_pil:
            base64_image = self.encode_pil_image_to_base64(image_pil)
            if messages and messages[-1]["role"] == "user":
                content = messages[-1]["content"]
                messages[-1]["content"] = [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        }
                    },
                    {
                        "type": "text",
                        "text": content
                    }
                ]
        
        if "o3" in self.model_name.lower():
            payload = {
                "model": self.model_name,
                "messages": messages,
                "max_completion_tokens": max_tokens
            }
        else:
            payload = {
                "model": self.model_name,
                "messages": messages,
                "max_tokens": max_tokens
            }
        
        try:
            response = requests.post(
                self.base_url, 
                headers=headers, 
                json=payload, 
                timeout=REQUEST_TIMEOUT
            )
            
            if response.status_code != 200:
                print(f"[O3Client] ❌ HTTP错误")
                print(f"  - 状态码: {response.status_code}")
                print(f"  - 响应文本: {response.text[:500]}")
                try:
                    error_json = response.json()
                    if 'error' in error_json:
                        print(f"  - 错误详情: {error_json['error']}")
                except:
                    pass
                return None
            
            try:
                result = response.json()
            except requests.exceptions.JSONDecodeError as e:
                print(f"[O3Client] ❌ 响应不是有效的JSON格式")
                print(f"  - 错误: {e}")
                print(f"  - 响应文本: {response.text[:500]}")
                return None
            
            if 'choices' in result and len(result['choices']) > 0:
                message = result['choices'][0].get('message', {})
                prediction = message.get('content', None)  # 使用.get()避免KeyError
                
                if prediction is not None and len(str(prediction).strip()) > 0:
                    return prediction
                else:
                    print(f"[O3Client] ❌ API返回的content为None或空")
                    print(f"  - choices[0].message: {message}")
                    print(f"  - finish_reason: {result['choices'][0].get('finish_reason', 'N/A')}")
                    
                    finish_reason = result['choices'][0].get('finish_reason', '')
                    if finish_reason == 'length':
                        print(f"  - ⚠️  原因: 生成内容超过最大长度限制，被截断")
                        print(f"  - 建议: 增加max_tokens参数或简化prompt")
                    
                    return None
            elif 'error' in result:
                error_info = result['error']
                print(f"[O3Client] ❌ API返回错误")
                print(f"  - 错误类型: {error_info.get('type', 'N/A')}")
                print(f"  - 错误码: {error_info.get('code', 'N/A')}")
                print(f"  - 错误信息: {error_info.get('message', 'N/A')}")
                print(f"  - 完整错误: {error_info}")
                return None
            else:
                print(f"[O3Client] ❌ API返回格式异常（无choices或error字段）")
                print(f"  - 完整响应: {result}")
                return None
                
        except requests.exceptions.Timeout as e:
            print(f"[O3Client] ❌ 请求超时")
            print(f"  - 超时时间: {REQUEST_TIMEOUT}秒")
            print(f"  - 错误: {str(e)}")
            return None
        except requests.exceptions.ConnectionError as e:
            print(f"[O3Client] ❌ 连接错误")
            print(f"  - 错误: {str(e)}")
            return None
        except Exception as e:
            print(f"[O3Client] ❌ API调用异常")
            print(f"  - 异常类型: {type(e).__name__}")
            print(f"  - 错误信息: {str(e)}")
            try:
                if 'response' in locals():
                    print(f"  - HTTP状态码: {response.status_code}")
                    print(f"  - 响应文本: {response.text[:300]}")
            except:
                pass
            return None


class SerperClient:
    
    def __init__(self, tmp_dir: str = TMP_IMAGE_DIR,
                 min_resolution: int = 512,  
                 min_sharpness: float = 80.0,  
                 min_size_kb: int = 20,  
                 max_size_mb: float = 5.0):  

        self.api_key = SERPER_API_KEY
        self.tmp_dir = tmp_dir
        self.min_resolution = min_resolution
        self.min_sharpness = min_sharpness
        self.min_size_kb = min_size_kb
        self.max_size_mb = max_size_mb
        os.makedirs(self.tmp_dir, exist_ok=True)
    
    def _calculate_sharpness(self, image: Image.Image) -> float:

        try:
            gray = image.convert('L')
            img_array = np.array(gray)
            
            laplacian = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
            
            from scipy.ndimage import convolve
            edges = convolve(img_array.astype(float), laplacian)
            
            sharpness = edges.var()
            return sharpness
            
        except ImportError:
            try:
                gray = image.convert('L')
                img_array = np.array(gray).astype(float)
                
                dx = np.diff(img_array, axis=1)
                dy = np.diff(img_array, axis=0)
                
                sharpness = np.var(dx) + np.var(dy)
                return sharpness
            except:
                return 100.0
        except Exception:
            return 100.0  
    
    def _check_image_quality(self, img: Image.Image, file_size_bytes: int) -> tuple[bool, str]:

        width, height = img.size
        file_size_kb = file_size_bytes / 1024
        file_size_mb = file_size_bytes / (1024 * 1024)
        
        if width < self.min_resolution and height < self.min_resolution:
            return False, f"分辨率过低 ({width}x{height} < {self.min_resolution})"
        
        if file_size_kb < self.min_size_kb:
            return False, f"文件过小 ({file_size_kb:.1f}KB)"
        
        if file_size_mb > self.max_size_mb:
            return False, f"文件过大 ({file_size_mb:.1f}MB)"
        
        sharpness = self._calculate_sharpness(img)
        if sharpness < self.min_sharpness:
            return False, f"清晰度不足 (分数:{sharpness:.1f} < {self.min_sharpness})"
        
        return True, f"✓ 质量良好 ({width}x{height}, {file_size_kb:.1f}KB, 清晰度:{sharpness:.1f})"
    
    def download_image(self, image_url: str, timeout: int = 10, check_quality: bool = True) -> Optional[str]:

        try:
            response = requests.get(
                image_url,
                timeout=timeout,
                headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
                },
                stream=True  
            )
            
            if response.status_code != 200:
                print(f"    ✗ 下载失败 (HTTP {response.status_code})")
                return None
            
            content = response.content
            file_size_bytes = len(content)
            image_data = BytesIO(content)
            
            try:
                img = Image.open(image_data)
                img.verify()  
                
                image_data.seek(0)
                img = Image.open(image_data)
                
                if img.mode not in ('RGB', 'RGBA'):
                    img = img.convert('RGB')
                
                if check_quality:
                    passed, reason = self._check_image_quality(img, file_size_bytes)
                    if not passed:
                        print(f"    ✗ 质量检测未通过: {reason}")
                        return None
                    else:
                        width, height = img.size
                        print(f"    {reason}")
                
                filename = f"serper_{uuid.uuid4().hex[:12]}.jpg"
                filepath = os.path.join(self.tmp_dir, filename)
                
                img.save(filepath, format='JPEG', quality=95)
                
                if not os.path.exists(filepath):
                    print(f"    ✗ 文件保存失败")
                    return None
                
                file_size_kb = os.path.getsize(filepath) / 1024
                print(f"    ✓ 已保存 → {filename}")
                return filepath
                
            except Exception as e:
                print(f"    ✗ 图片处理失败: {str(e)}")
                return None
        
        except requests.exceptions.Timeout:
            print(f"    ✗ 下载超时")
            return None
        except requests.exceptions.ConnectionError:
            print(f"    ✗ 连接失败")
            return None
        except Exception as e:
            print(f"    ✗ 下载异常: {type(e).__name__} - {str(e)}")
            return None
    
    def search_images(self, query: str, num_results: int = 3, 
                     page: int = 1,
                     download: bool = True) -> List[Dict[str, str]]:

        api_url = "https://google.serper.dev/images"
        
        payload = {
            "q": query,
            "page": page,
            "num": 10  
        }
        
        headers = {
            'X-API-KEY': self.api_key,
            'Content-Type': 'application/json'
        }
        
        try:
            print(f"[SerperClient] 搜索图片: '{query}' (页{page})")
            response = requests.post(
                api_url, 
                headers=headers, 
                json=payload,
                timeout=REQUEST_TIMEOUT
            )
            
            if response.status_code != 200:
                print(f"[SerperClient] ❌ HTTP错误")
                print(f"  - 状态码: {response.status_code}")
                print(f"  - 响应: {response.text[:500]}")
                return []
            
            results = response.json()
            
            if "images" not in results:
                print(f"[SerperClient] 搜索无结果: {query}")
                return []
            
            candidates = []
            for img in results["images"]:
                url = img.get("imageUrl", "") 
                if url: 
                    candidates.append({
                        "url": url,
                        "title": img.get("title", ""),
                        "source": img.get("source", ""),
                        "domain": img.get("domain", "")
                    })
            
            print(f"[SerperClient] 获得 {len(candidates)} 个候选URL")
            
            if not download:
                return candidates[:num_results]
            
            print(f"[SerperClient] 开始下载图片...")
            downloaded_images = []
            
            for idx, img in enumerate(candidates):
                if len(downloaded_images) >= num_results:
                    break  
                
                url = img["url"]
                print(f"  [{idx+1}/{len(candidates)}] 下载: {url[:80]}...")
                
                local_path = self.download_image(url)
                if local_path:
                    downloaded_images.append({
                        "url": f"file://{local_path}", 
                        "local_path": local_path,      
                        "original_url": url,           
                        "title": img.get("title", ""),
                        "source": img.get("source", ""),
                        "domain": img.get("domain", "")
                    })
                    print(f"    成功 ({len(downloaded_images)}/{num_results})")
            
            print(f"[SerperClient] 成功下载 {len(downloaded_images)}/{num_results} 张图片")
            return downloaded_images
            
        except requests.exceptions.Timeout as e:
            print(f"[SerperClient] ❌ 请求超时")
            print(f"  - 超时时间: {REQUEST_TIMEOUT}秒")
            print(f"  - 错误: {str(e)}")
            time.sleep(2)
            return []
        except requests.exceptions.ConnectionError as e:
            print(f"[SerperClient] ❌ 连接错误")
            print(f"  - 错误: {str(e)}")
            time.sleep(2)
            return []
        except Exception as e:
            print(f"[SerperClient] 图片搜索失败")
            print(f"  - 查询: {query}")
            print(f"  - 异常类型: {type(e).__name__}")
            print(f"  - 错误信息: {str(e)}")
            time.sleep(2)  
            return []


class QwenImageEditClient:
    
    def __init__(self):
        self.api_key = DASHSCOPE_API_KEY
        dashscope.base_http_api_url = DASHSCOPE_IMAGE_EDIT_URL
    
    def download_image(self, image_url: str) -> Optional[Image.Image]:
        try:
            response = requests.get(image_url, stream=True, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content))
            return image
        except Exception as e:
            print(f"图片下载失败 {image_url}: {e}")
            return None
    
    def verify_url_accessible(self, image_url: str) -> bool:

        if image_url.startswith("file://"):
            return True
        
        try:
            response = requests.head(image_url, timeout=10, allow_redirects=True)
            
            if response.status_code == 405:
                response = requests.get(image_url, stream=True, timeout=10)
                response.close()
            
            if response.status_code == 200:
                return True
            elif response.status_code == 403:
                print(f"[QwenImageEditClient] URL访问被拒绝(403): {image_url[:100]}...")
                return False
            elif response.status_code == 404:
                print(f"[QwenImageEditClient] URL不存在(404): {image_url[:100]}...")
                return False
            else:
                print(f"[QwenImageEditClient] URL返回异常状态码({response.status_code}): {image_url[:100]}...")
                return False
                
        except requests.exceptions.Timeout:
            print(f"[QwenImageEditClient] URL访问超时: {image_url[:100]}...")
            return False
        except requests.exceptions.ConnectionError:
            print(f"[QwenImageEditClient] URL连接失败: {image_url[:100]}...")
            return False
        except Exception as e:
            print(f"[QwenImageEditClient] URL验证异常: {image_url[:100]}...")
            print(f"  - 错误: {str(e)}")
            return False
    
    def edit_images(self, instruction: str, reference_images: List[str], 
                    n: int = 2, 
                    negative_prompt: str = "低质量, 模糊, 失真, 不自然的融合, 边界不清晰, 色彩不协调, 光影不匹配, 扭曲变形, 锯齿边缘, 低分辨率, 噪点, 过度曝光, 欠曝光",
                    prompt_extend: bool = True) -> List[str]:

        print(f"[QwenImageEditClient] 验证 {len(reference_images[:3])} 张参考图片URL可访问性...")
        accessible_urls = []
        for idx, img_url in enumerate(reference_images[:3]):
            print(f"  验证参考图 {idx+1}: {img_url[:100]}...")
            if self.verify_url_accessible(img_url):
                accessible_urls.append(img_url)
                print(f"    ✓ 可访问")
            else:
                print(f"    ✗ 不可访问，将跳过此图片")
        
        if not accessible_urls:
            print(f"[QwenImageEditClient] 错误: 所有参考图片URL都不可访问")
            return []
        
        print(f"[QwenImageEditClient] 使用 {len(accessible_urls)} 张可访问的参考图生成")
        
        messages = [
            {
                "role": "user",
                "content": []
            }
        ]
        
        for idx, img_url in enumerate(accessible_urls):
            if img_url.startswith("file://"):
                local_path = img_url.replace("file://", "")
                try:
                    with open(local_path, "rb") as f:
                        img_data = f.read()
                        base64_img = base64.b64encode(img_data).decode('utf-8')
                        data_uri = f"data:image/jpeg;base64,{base64_img}"
                        messages[0]["content"].append({"image": data_uri})
                        print(f"  参考图 {idx+1}: {local_path} (已转换为base64)")
                except Exception as e:
                    print(f"  参考图 {idx+1}: 读取失败 - {e}")
                    continue
            else:
                messages[0]["content"].append({"image": img_url})
                print(f"  参考图 {idx+1}: {img_url[:100]}...")
        
        messages[0]["content"].append({"text": instruction})
        print(f"[QwenImageEditClient] 编辑指令: {instruction}")
        
        try:
            response = MultiModalConversation.call(
                api_key=self.api_key,
                model="qwen-image-edit-plus",
                messages=messages,
                stream=False,
                n=n,
                watermark=False,
                negative_prompt=negative_prompt,
                prompt_extend=prompt_extend,
            )
            
            if response.status_code == 200:
                image_urls = []
                for content in response.output.choices[0].message.content:
                    image_urls.append(content['image'])
                print(f"[QwenImageEditClient] 成功生成 {len(image_urls)} 张图片")
                return image_urls
            else:
                print(f"[QwenImageEditClient] 图片生成失败")
                print(f"  - HTTP状态码: {response.status_code}")
                print(f"  - 错误消息: {response.message}")
                print(f"  - 错误码: {getattr(response, 'code', 'N/A')}")
                time.sleep(2)  
                return []
        except Exception as e:
            print(f"[QwenImageEditClient] 图片生成异常")
            print(f"  - 异常类型: {type(e).__name__}")
            print(f"  - 错误信息: {str(e)}")
            time.sleep(2)  
            return []


class Qwen3VLClient:
    
    def __init__(self, model_name: str = "qwen3-vl-plus"):
        from openai import OpenAI
        
        self.model_name = model_name
        self.client = OpenAI(
            api_key=DASHSCOPE_API_KEY,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
    
    def _compress_image_if_needed(self, image: Image.Image, max_size_mb: float = 9.5) -> Image.Image:

        max_size_bytes = int(max_size_mb * 1024 * 1024)
        
        buffer = BytesIO()
        image.save(buffer, format='JPEG', quality=95)
        current_size = buffer.tell()
        
        if current_size <= max_size_bytes:
            return image
        
        print(f"[Qwen3VLClient] ⚠️  图片过大 ({current_size / 1024 / 1024:.2f}MB)，开始压缩...")
        
        original_width, original_height = image.size
        scale_factor = (max_size_bytes / current_size) ** 0.5
        
        scale_factor = max(0.3, min(0.7, scale_factor))
        
        new_width = int(original_width * scale_factor)
        new_height = int(original_height * scale_factor)
        
        compressed_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        for quality in [85, 75, 65, 55, 45]:
            buffer = BytesIO()
            compressed_image.save(buffer, format='JPEG', quality=quality)
            final_size = buffer.tell()
            
            if final_size <= max_size_bytes:
                print(f"[Qwen3VLClient] ✓ 压缩成功: {original_width}x{original_height} -> {new_width}x{new_height}, "
                      f"{current_size / 1024 / 1024:.2f}MB -> {final_size / 1024 / 1024:.2f}MB (质量={quality})")
                return compressed_image
        
        print(f"[Qwen3VLClient] ⚠️  继续缩小分辨率...")
        scale_factor = 0.5
        new_width = int(new_width * scale_factor)
        new_height = int(new_height * scale_factor)
        compressed_image = compressed_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        buffer = BytesIO()
        compressed_image.save(buffer, format='JPEG', quality=75)
        final_size = buffer.tell()
        
        print(f"[Qwen3VLClient] ✓ 最终压缩: {original_width}x{original_height} -> {new_width}x{new_height}, "
              f"{current_size / 1024 / 1024:.2f}MB -> {final_size / 1024 / 1024:.2f}MB")
        
        return compressed_image
    
    def encode_pil_image_to_base64(self, image: Image.Image) -> str:
        if image.mode not in ('RGB', 'L'):
            image = image.convert('RGB')
        
        compressed_image = self._compress_image_if_needed(image)
        
        buffer = BytesIO()
        compressed_image.save(buffer, format='JPEG', quality=95)
        img_str = base64.b64encode(buffer.getvalue()).decode()
        
        return img_str
    
    def chat(self, messages: List[Dict], image_url: str = None, image_path: str = None, 
             image_pil: Image.Image = None, max_tokens: int = 2000) -> Optional[str]:

        try:
            if len(messages) > 0 and isinstance(messages[0].get("content"), str):
                text_content = messages[0]["content"]
                content_list = []
                
                if image_url:
                    content_list.append({
                        "type": "image_url",
                        "image_url": {"url": image_url}
                    })
                elif image_path:
                    img = Image.open(image_path)
                    if img.mode not in ('RGB', 'L'):
                        img = img.convert('RGB')
                    compressed_img = self._compress_image_if_needed(img)
                    buffer = BytesIO()
                    compressed_img.save(buffer, format='JPEG', quality=95)
                    img_base64 = base64.b64encode(buffer.getvalue()).decode()
                    content_list.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}
                    })
                elif image_pil:
                    img_base64 = self.encode_pil_image_to_base64(image_pil)
                    content_list.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}
                    })
                
                content_list.append({
                    "type": "text",
                    "text": text_content
                })
                
                messages = [
                    {
                        "role": "user",
                        "content": content_list
                    }
                ]
            
            print(f"[Qwen3VLClient] 调用 {self.model_name} API...")
            
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=max_tokens
            )
            
            if completion.choices and len(completion.choices) > 0:
                content = completion.choices[0].message.content
                print(f"[Qwen3VLClient] API调用成功，响应长度: {len(content)} 字符")
                return content
            else:
                print(f"[Qwen3VLClient] API响应为空")
                if hasattr(completion, 'error'):
                    print(f"  - 错误信息: {completion.error}")
                return None
                
        except Exception as e:
            print(f"[Qwen3VLClient] API调用失败")
            print(f"  - 异常类型: {type(e).__name__}")
            print(f"  - 错误信息: {str(e)}")
            return None


