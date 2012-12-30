module fields
    use utils
    implicit none

contains

subroutine grad_1d(a, g, dx)
    real, intent(in) :: a(:), dx
    real, intent(out) :: g(1, size(a, 1))
    integer :: x

    do x = 1, size(a, 1)
        g(1, x, y) = a(i_wrap(x + 1, size(a, 1))) - a(i_wrap(x - 1, size(a, 1)))
    end do
    g = g / (2.0 * dx)
end subroutine

subroutine grad_2d(a, g, dx)
    real, intent(in) :: a(:, :), dx
    real, intent(out) :: g(2, size(a, 1), size(a, 2))
    integer :: x, y

    do y = 1, size(a, 2)
        do x = 1, size(a, 1)
            g(1, x, y) = a(i_wrap(x + 1, size(a, 1)), y) - a(i_wrap(x - 1, size(a, 1)), y)
            g(2, x, y) = a(x, i_wrap(y + 1, size(a, 2))) - a(x, i_wrap(y - 1, size(a, 2)))
        end do
    end do
    g = g / (2.0 * dx)
end subroutine

subroutine div_2d(a, d, dx)
    real, intent(in) :: a(2, :, :), dx
    real, intent(out) :: d(size(a, 2), size(a, 3))
    integer :: x, y

    do y = 1, size(a, 3)
        do x = 1, size(a, 2)
            d(x, y) = (a(1, i_wrap(x + 1, size(a, 2)), y) - a(1, i_wrap(x - 1, size(a, 2)), y)) &
                      + (a(2, x, i_wrap(y + 1, size(a, 3))) - a(2, x, i_wrap(y - 1, size(a, 3))))
        end do
    end do
    d = d / (2.0 * dx)
end subroutine

subroutine laplacian_1d(a, l, dx)
    real, intent(in) :: a(:), dx
    real, intent(out) :: l(size(a, 1))
    integer :: x

    do x = 1, size(a, 1)
        l(x) = a(i_wrap(x + 1, size(a, 1))) + a(i_wrap(x - 1, size(a, 1)))
    end do
    l = (l - 2.0 * a) / (dx ** 2.0)
end subroutine

subroutine laplacian_2d(a, l, dx)
    real, intent(in) :: a(:, :), dx
    real, intent(out) :: l(size(a, 1), size(a, 2))
    integer :: x, y

    do y = 1, size(a, 2)
        do x = 1, size(a, 1)
            l(x, y) = a(x, i_wrap(y + 1, size(a, 2))) + a(x, i_wrap(y - 1, size(a, 2))) &
                      + a(i_wrap(x + 1, size(a, 1)), y) + a(i_wrap(x - 1, size(a, 1)), y)
        end do
    end do
    l = (l - 4.0 * a) / (dx ** 2.0)
end subroutine

end module
