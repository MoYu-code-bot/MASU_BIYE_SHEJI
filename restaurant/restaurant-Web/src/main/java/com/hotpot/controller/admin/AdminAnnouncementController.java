package com.hotpot.controller.admin;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.hotpot.common.Result;
import com.hotpot.entity.Announcement;
import com.hotpot.service.AnnouncementService;
import io.swagger.annotations.Api;
import io.swagger.annotations.ApiOperation;
import io.swagger.annotations.ApiParam;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@Api(tags = "B端-公告管理")
@RestController
@RequestMapping("/admin/announcements")
@RequiredArgsConstructor
public class  AdminAnnouncementController {

    private final AnnouncementService announcementService;

    @ApiOperation("查询公告列表")
    @GetMapping
    public Result<List<Announcement>> list(@ApiParam("门店ID") @RequestParam(required = false) Long storeId) {
        if (storeId != null) {
            return Result.success(announcementService.list(
                    new LambdaQueryWrapper<Announcement>()
                            .eq(Announcement::getStoreId, storeId)
                            .orderByDesc(Announcement::getCreateTime)));
        }
        return Result.success(announcementService.list());
    }

    @ApiOperation("新增公告")
    @PostMapping
    public Result<?> add(@ApiParam("公告信息") @RequestBody Announcement announcement) {
        announcementService.save(announcement);
        return Result.success();
    }

    @ApiOperation("修改公告")
    @PutMapping("/update")
    public Result<?> update(@ApiParam("公告ID") @RequestParam Long announcementId,
                            @ApiParam("公告信息") @RequestBody Announcement announcement) {
        announcement.setId(announcementId);
        announcementService.updateById(announcement);
        return Result.success();
    }

    @ApiOperation("删除公告")
    @DeleteMapping("/delete")
    public Result<?> delete(@ApiParam("公告ID") @RequestParam Long announcementId) {
        announcementService.removeById(announcementId);
        return Result.success();
    }
}
