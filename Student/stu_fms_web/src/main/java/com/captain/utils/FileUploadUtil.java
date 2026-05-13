package com.captain.utils;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.web.multipart.MultipartFile;
import java.io.File;
import java.util.UUID;

/**
 * 文件上传工具
 */
public class FileUploadUtil {
    private static final Logger logger= LoggerFactory.getLogger(FileUploadUtil.class);

    public static String getVirtualRootPath(){
        return "/upload/";
    }

    /**
     * 获取真实的文件保存路径根目录
     * @return
     */
    public static String getRealRootPath(){
        //判断当前操作系统环境,返回不同的根目录
        String osName=System.getProperty("os.name").toLowerCase();
        if(osName.equals("linux")){
            return "/stu_fms/upload/";
        }else{
            return "C:/stu_fms/upload/";
        }
    }

    /**
     * 保存图片
     * @param multipartFile
     * @return
     */
    public static String saveImage(MultipartFile multipartFile){
        //获取文件的后缀名
        String originalName=multipartFile.getOriginalFilename();
        String ext=originalName.substring(originalName.indexOf('.')+1).toLowerCase();
        //创建新名字
        String filename= UUID.randomUUID().toString().toLowerCase()+"."+ext;
        //创建文件
        File file=new File(getRealRootPath()+"/images/"+filename);
        try{
            multipartFile.transferTo(file);
            return "/api/"+getVirtualRootPath()+"/images/"+filename;
        }catch (Exception e){
            logger.error("保存图片失败");
            return null;
        }
    }

}
