package com.captain.controller;

import com.captain.common.Result;
import com.captain.utils.FileUploadUtil;
import org.springframework.util.Assert;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

/**
 * 文件上传下载Controller
 * @author  lainhong
 */
@RestController
public class FileController {

    /**
     * 上传图片
     * @param file
     * @return 访问的虚拟路径
     */
    @PostMapping("/upload/img")
    public Result uploadImage(MultipartFile file){
        Assert.notNull(file,"参数不能为空");
        String path = FileUploadUtil.saveImage(file);
        if(path!=null){
            return Result.success(path);
        }
        return Result.fail();
    }

}
