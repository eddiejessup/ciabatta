program control_file_example
    implicit none

    ! Input related variables
    character(len=100), parameter :: f_in_name = 'params_file_example.txt'
    integer, parameter :: f_in_unit = 15

    ! Parameters
    real :: pi

    call parse_params()

contains

subroutine parse_params()
    character(len=100) :: buffer, label
    integer :: pos, ios = 0, lines = 0

    open(unit=f_in_unit, file=f_in_name)
    do while (.true.)
        read(f_in_unit, '(a)', iostat=ios) buffer
        if (ios == 0) then
            pos = scan(buffer, ' 	')
            label = buffer(1:pos)
            buffer = buffer(pos + 1:)
            select case (label)
            case ('pi')
                read(buffer, *, iostat=ios) pi
                print *, 'pi', pi
            case default
                print *, 'invalid line found, ignoring'
                lines = lines - 1
            end select
            lines = lines + 1
        else
            exit
        end if
    end do
    if (lines == 1) then
        print *, 'all variables defined, continuing...'
    else
        print *, 'not all variables defined, stopping...'
        stop
    end if
end subroutine

end program
