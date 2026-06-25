package com.hotpot.controller.admin;

import com.hotpot.common.Result;
import com.hotpot.entity.Banner;
import com.hotpot.service.BannerService;
import io.swagger.annotations.Api;
import io.swagger.annotations.ApiImplicitParam;
import io.swagger.annotations.ApiOperation;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@Api(tags = "B端-轮播图管理")
@RestController
@RequestMapping("/admin/banners")
@RequiredArgsConstructor
public class AdminBannerController {

    private final BannerService bannerService;

    @ApiOperation("查询全部轮播图")
    @GetMapping
    public Result<List<Banner>> list() {
        return Result.success(bannerService.list());
    }

    @ApiOperation("新增轮播图")
    @PostMapping
    public Result<?> add(@RequestBody Banner banner) {
        bannerService.save(banner);
        return Result.success();
    }

    @ApiOperation("修改轮播图")
    @ApiImplicitParam(name = "bannerId", value = "轮播图ID", required = true, dataType = "long", paramType = "query")
    @PutMapping("/update")
    public Result<?> update(@RequestParam Long bannerId, @RequestBody Banner banner) {
        banner.setId(bannerId);
        bannerService.updateById(banner);
        return Result.success();
    }

    @ApiOperation("删除轮播图")
    @ApiImplicitParam(name = "bannerId", value = "轮播图ID", required = true, dataType = "long", paramType = "query")
    @DeleteMapping("/delete")
    public Result<?> delete(@RequestParam Long bannerId) {
        bannerService.removeById(bannerId);
        return Result.success();
    }
}
