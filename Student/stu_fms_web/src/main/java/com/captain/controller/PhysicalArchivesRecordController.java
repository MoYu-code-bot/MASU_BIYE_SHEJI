package com.captain.controller;

import com.baomidou.mybatisplus.core.conditions.query.QueryWrapper;
import com.captain.common.BaseRestController;
import com.captain.common.Result;
import com.captain.entity.po.PhysicalBorrRecord;
import com.captain.service.PhysicalBorrRecordService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.util.Assert;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;

@RestController
@RequestMapping("/physical/archives/record")
public class PhysicalArchivesRecordController extends BaseRestController<PhysicalBorrRecordService, PhysicalBorrRecord> {

    @Autowired
    private PhysicalBorrRecordService physicalBorrRecordService;

    @Override
    protected QueryWrapper<PhysicalBorrRecord> getSearchQueryWrapper(String searchValue, PhysicalBorrRecord physicalBorrRecord) {

        QueryWrapper<PhysicalBorrRecord> physicalBorrRecordQueryWrapper = new QueryWrapper<PhysicalBorrRecord>();
        physicalBorrRecordQueryWrapper.like("username", searchValue);
        return physicalBorrRecordQueryWrapper;
    }

    //通过用户id查询他的所有记录
    @GetMapping("/getOne")
    public Result getByUsernameId(Long usernameId) {
        Assert.notNull(usernameId, "id为空");
        List<PhysicalBorrRecord> list = physicalBorrRecordService.findOneByusernameID(usernameId);
        System.out.println(list);
        return Result.success(list);
    }
}
