package com.captain.controller;

import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.captain.common.BaseRestController;
import com.captain.common.Result;
import com.captain.entity.po.PhysicalArchives;
import com.captain.service.PhysicalArchivesService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.util.Assert;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/physical/archives")
public class PhysicalArchivesController extends BaseRestController<PhysicalArchivesService,PhysicalArchives>{


    @Autowired
    private PhysicalArchivesService physicalArchivesService;
    //出库
    @GetMapping("/outStock")
    public Result outStock(Long id){
        Assert.notNull(id,"id不能为空");
        boolean b = physicalArchivesService.outStock(id);

        if(b){
            return Result.success();
        }
        else {
            return Result.fail();
        }
    }

    //查询借阅

    /**
     *
     * @param page
     * @param size
     * @return
     */
    @GetMapping("/borrow/list")
    public Result getBorrowList(Integer page,Integer size){
        Assert.notNull(page, "页数不能为空");
        Assert.notNull(size, "页大小不能为空");
        IPage<PhysicalArchives> list = physicalArchivesService.findListByUsernameID(new Page(page,size));
        System.out.println("list="+list);
        return Result.success(list);
    }


    //查询
}
