package com.hotpot.controller.admin;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.hotpot.common.PageResult;
import com.hotpot.common.Result;
import com.hotpot.dto.PageQuery;
import com.hotpot.entity.Review;
import com.hotpot.service.ReviewService;
import io.swagger.annotations.Api;
import io.swagger.annotations.ApiImplicitParam;
import io.swagger.annotations.ApiOperation;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;

@Api(tags = "B端-评价管理")
@RestController
@RequestMapping("/admin/reviews")
@RequiredArgsConstructor
public class AdminReviewController {

    private final ReviewService reviewService;

    @ApiOperation("分页查询评价")
    @GetMapping
    public Result<PageResult<Review>> page(PageQuery query,
                                           @RequestParam(required = false) Long storeId) {
        Page<Review> page = new Page<>(query.getPageNum(), query.getPageSize());
        LambdaQueryWrapper<Review> wrapper = new LambdaQueryWrapper<>();
        if (storeId != null) {
            wrapper.eq(Review::getStoreId, storeId);
        }
        wrapper.orderByDesc(Review::getCreateTime);
        return Result.success(PageResult.of(reviewService.page(page, wrapper)));
    }

    @ApiOperation("修改评价可见状态")
    @PutMapping("/updateVisible")
    public Result<?> updateVisible(@RequestParam Long reviewId, @RequestParam Integer isVisible) {
        Review review = new Review();
        review.setId(reviewId);
        review.setIsVisible(isVisible);
        reviewService.updateById(review);
        return Result.success();
    }
}
