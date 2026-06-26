package com.hotpot.controller.admin;

import com.hotpot.common.PageResult;
import com.hotpot.common.Result;
import com.hotpot.dto.PageQuery;
import com.hotpot.entity.Dish;
import com.hotpot.service.DishService;
import io.swagger.annotations.Api;
import io.swagger.annotations.ApiOperation;
import io.swagger.annotations.ApiParam;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;

@Api(tags = "B端-菜品管理")
@RestController
@RequestMapping("/admin/dishes")
@RequiredArgsConstructor
public class AdminDishController {

    private final DishService dishService;

    @GetMapping("list")
    @ApiOperation("分页查询菜品")
    public Result<PageResult<Dish>> page(@ApiParam("分页参数") PageQuery query,
                                         @ApiParam("分类ID") @RequestParam(required = false) Long categoryId,
                                         @ApiParam("门店ID") @RequestParam(required = false) Long storeId,
                                         @ApiParam("状态") @RequestParam(required = false) Integer status) {
        return Result.success(dishService.pageQuery(query, categoryId, storeId, status));
    }

    @PostMapping("create")
    @ApiOperation("新增菜品")
    public Result<?> add(@ApiParam("菜品信息") @RequestBody Dish dish) {
        dishService.save(dish);
        return Result.success();
    }

    @PutMapping("update")
    @ApiOperation("修改菜品")
    public Result<?> update(@ApiParam("菜品ID") @RequestParam Long dishId,
                            @ApiParam("菜品信息") @RequestBody Dish dish) {
        dish.setId(dishId);
        dishService.updateById(dish);
        return Result.success();
    }

    @PutMapping("updateStatus")
    @ApiOperation("菜品上下架")
    public Result<?> updateStatus(@ApiParam("菜品ID") @RequestParam Long dishId,
                                  @ApiParam("状态：1-上架，0-下架") @RequestParam Integer status) {
        Dish dish = new Dish();
        dish.setId(dishId);
        dish.setIsOnSale(status);
        dishService.updateById(dish);
        return Result.success();
    }

    @DeleteMapping("delete")
    @ApiOperation("删除菜品")
    public Result<?> delete(@ApiParam("菜品ID") @RequestParam Long dishId) {
        dishService.removeById(dishId);
        return Result.success();
    }
}
