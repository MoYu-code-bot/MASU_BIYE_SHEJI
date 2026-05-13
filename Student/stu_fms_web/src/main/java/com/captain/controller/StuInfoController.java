package com.captain.controller;

import com.baomidou.mybatisplus.core.conditions.query.QueryWrapper;
import com.captain.common.BaseRestController;
import com.captain.common.Result;
import com.captain.entity.po.StuInfo;
import com.captain.service.StuInfoService;
import org.apache.shiro.util.Assert;
import org.springframework.util.StringUtils;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

/**
 * 学生基本信息表
 */
@RestController
@RequestMapping("/stu_info")
public class StuInfoController extends BaseRestController<StuInfoService, StuInfo> {

    /**
     *通过用户ID来获取对应的学生信息
     * @return
     */
    @GetMapping("/getByUid")
    public Result getByUid(Long uid){
        Assert.notNull(uid,"用户ID不能为空");
        StuInfo stuInfo = baseService.getOne(new QueryWrapper<StuInfo>()
                .eq("uid", uid));
        return Result.success(stuInfo);
    }

    /**
     * 通过学生ID来获取数据
     * @return
     */
    @GetMapping("/getByStuId")
    public Result getByStuId(String stuId){
        Assert.notNull(stuId,"参数不能为空");
        StuInfo stuInfo = baseService.getOne(new QueryWrapper<StuInfo>()
                .eq("stu_id", stuId));
        if(stuInfo!=null){
            return Result.success(stuInfo);
        }
        return Result.fail("不存在学号为【"+stuId+"】的学生信息");
    }

    @Override
    protected QueryWrapper<StuInfo> getSearchQueryWrapper(String searchValue, StuInfo stuInfo) {
        if(!StringUtils.isEmpty(searchValue)){
            QueryWrapper<StuInfo> wrapper=new QueryWrapper();
            wrapper.like("stu_id",searchValue);
            return wrapper;
        }else{
            return super.getSearchQueryWrapper(searchValue, stuInfo);
        }
    }
}
