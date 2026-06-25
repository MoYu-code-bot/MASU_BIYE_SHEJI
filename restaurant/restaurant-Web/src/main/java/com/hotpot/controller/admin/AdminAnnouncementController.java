package com.hotpot.controller.admin;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.hotpot.common.Result;
import com.hotpot.entity.Announcement;
import com.hotpot.service.AnnouncementService;
import io.swagger.annotations.Api;
import io.swagger.annotations.ApiImplicitParam;
import io.swagger.annotations.ApiOperation;
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
    public Result<List<Announcement>> list(@RequestParam(required = false) Long storeId) {
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
    public Result<?> add(@RequestBody Announcement announcement) {
        announcementService.save(announcement);
        return Result.success();
    }

    @ApiOperation("修改公告")
    @ApiImplicitParam(name = "announcementId", value = "公告ID", required = true, dataType = "long", paramType = "query")
    @PutMapping("/update")
    public Result<?> update(@RequestParam Long announcementId, @RequestBody Announcement announcement) {
        announcement.setId(announcementId);
        announcementService.updateById(announcement);
        return Result.success();
    }

    @ApiOperation("删除公告")
    @ApiImplicitParam(name = "announcementId", value = "公告ID", required = true, dataType = "long", paramType = "query")
    @DeleteMapping("/delete")
    public Result<?> delete(@RequestParam Long announcementId) {
        announcementService.removeById(announcementId);
        return Result.success();
    }
}
